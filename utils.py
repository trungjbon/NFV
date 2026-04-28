import pandas as pd
from entity import *
from time import sleep
import json
import copy
import random
import time
import numpy as np

SEED = 42

def set_seed(seed_value):
    # 1. Set seed cho thư viện random chuẩn của Python
    random.seed(seed_value)
    
    # 2. Set seed cho thư viện NumPy (thường dùng trong tính toán fitness/surrogate)
    np.random.seed(seed_value)
    
    print(f"--- Seed đã được thiết lập: {seed_value} ---")

set_seed(SEED)

def read_vnfs(data):
    vnfs = {}

    for i, value in enumerate(data["F"]):
        cpu_demand = value["c_f"]
        ram_demand = value["r_f"]
        mem_demand = value["h_f"]
        boot_time = {}
        for key, value in value["d_f"].items():
            boot_time[int(key)] = value

        vnfs[i] = VNF(i, cpu_demand, ram_demand, mem_demand, boot_time)

    return vnfs

def read_requests(data, vnfs):
    requests = []
        
    for i, value in enumerate(data["R"]):
        arrival_time = value["T"]
        src_node = value["st_r"]
        dst_node = value["d_r"]
        sfc = []
        for name in value["F_r"]:
            sfc.append(vnfs[name]) 
        bandwidth = value["b_r"]
        due_date = value["d_max"] + arrival_time
        
        requests.append(
            Request(i, arrival_time, src_node, dst_node, sfc, bandwidth, due_date)    
        )
        
    return requests
    
def read_network(data, vnfs):
    total_capacity = 0
    total_bandwidth = 0
    nodes = {}
    graph = Graph(directed=False)

    for key, value in data["V"].items():
        name = int(key)
        if (value["server"] == True):
            capacity = {
                "cpu": value["c_v"],
                "ram": value["r_v"],
                "mem": value["h_v"]
            }
            delay = value["d_v"]
            cost = {
                "cpu": value["cost_c"],
                "ram": value["cost_r"],
                "mem": value["cost_h"]
            }
            node = Node(name, "server", delay, capacity, cost)  
            node.vnf_dict = {vnf: -1 for vnf in vnfs.values()}
            node.vnf_valid_time = {vnf: float("inf") for vnf in vnfs.values()}
            total_capacity += sum(capacity.values())
        else:
            capacity = {
                "cpu": 0,
                "ram": 0,
                "mem": 0
            }
            delay = 0
            cost = {
                "cpu": 0,
                "ram": 0,
                "mem": 0
            }
            node = Node(name, "switch", delay, capacity, cost)

        nodes[name] = node

    for value in data["E"]:
        u = nodes[int(value["u"])]
        v = nodes[int(value["v"])]
        bandwidth = value["b_l"]
        delay = value["d_l"]

        graph.add_edge(u, v, bandwidth, delay)
        total_bandwidth += bandwidth

    return graph, nodes, total_capacity, total_bandwidth

def read_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    vnfs = read_vnfs(data)
    graph, nodes, total_capacity, total_bandwidth = read_network(data, vnfs)
    requests = read_requests(data, vnfs)

    return graph, nodes, requests, total_capacity, total_bandwidth

##########################################################################################
def get_action_terminals(req_state, time, num_req_process, remain_total_capacity, remain_total_bandwidth,
                        len_active_reqs, total_capacity, total_bandwidth):
    req = req_state["Request"]
    vnf = req.sfc[req_state["Current_vnf"]]

    RDD = req.due_date - time
    NRP = (num_req_process + 1) / len_active_reqs
    RC = (remain_total_capacity - vnf.get_total_demand_capacity()) / total_capacity
    RB = (remain_total_bandwidth - req.bandwidth) / total_bandwidth

    return RDD, NRP, RC, RB

def get_order_terminals(req_state):
    req = req_state["Request"]
    vnf = req.sfc[req_state["Current_vnf"]]

    BW = req.bandwidth
    DC = vnf.get_total_demand_capacity()
    RSFCL = len(req.sfc) - req_state["Current_vnf"]

    return BW, DC, RSFCL

def get_placement_terminals(node, graph, vnf, time):
    NU = node.get_total_used_capacity() / node.get_total_capacity()
    if (node.vnf_dict[vnf] == -1):
        ND = node.delay + vnf.boot_time[node.name] # if (vnf.boot_time is not None) else 0)
    else:
        ND = node.delay + max(node.vnf_valid_time[vnf] - time, 0)
    NC = node.get_total_cost(vnf)
    NN = len(graph.neighbors(node))

    return NU, ND, NC, NN

def get_routing_terminals(path):    # node, target, time
    # Thêm xét thời gian node khả dụng
    # links_delay = sum(link.delay for link in path)
    # node_delay = path[-1].v.delay if (path) else 0
    # PD = links_delay + max(node.vnf_valid_time[target] - time, 0) + node_delay

    PD = sum(link.delay for link in path)
    if (path):
        MLU = max(link.get_link_utilization() for link in path)
    else:
        MLU = 0
    PL = len(path)

    return PD, MLU, PL

def evaluate(graph, nodes, requests, 
             start_t, end_t,
             total_capacity, total_bandwidth,
             action_rule, order_rule, placement_rule, routing_rule,
             display=False, threshold=0, collect_situation=False):
    
    act_data = []
    ord_data = []
    place_data = []
    route_data = []
    situation_act_id = [0]
    situation_ord_id = [0]
    situation_place_id = [0]
    situation_route_id = [0]

    time = start_t
    mlu = [0]
    total_cost = [0]
    remain_total_capacity = [total_capacity]
    remain_total_bandwidth = [total_bandwidth]
    server_node = [n for n in nodes.values() if (n.type == "server")]

    if (display):
        print("Before:")
        print("Total capacity:", total_capacity)
        print("Total bandwidth:", total_bandwidth)
        print("Remain total capacity:", remain_total_capacity)
        print("Remain total bandwidth:", remain_total_bandwidth)

    active_requests = []
    finished_requests = []
    rejected_requests = []

    def request_coming():
        for r in requests:
            if (r.arrival_time == time):
                ingress = VNF(name="i_" + str(r.name))
                egress = VNF(name="e_" + str(r.name))

                active_requests.append({
                    "Request": r,
                    "Current_vnf": 0,
                    "Current_location": nodes[r.src_node],
                    "Placement": [],
                    "Path": [],
                    "Finish_time": float("inf"),
                    "Cost": 0,
                    "Process_time": 0,
                    "Using": [(nodes[r.src_node], ingress)],
                    "Ingress": ingress,
                    "Egress": egress
                })

                nodes[r.src_node].deploy(ingress, time)
                nodes[r.src_node].using(ingress)

    # ==== Rule 1 ======================================================
    def action_request():
        process_requests = []
        num_req_process = sum(1 for req_state in active_requests if (req_state["Process_time"] > 0))
        
        if (collect_situation and (time in range(start_t, end_t + 1))):
            situation_act_id[0] += 1

        for req_state in active_requests:
            if ((req_state["Process_time"] > 0) or (req_state["Current_vnf"] >= len(req_state["Request"].sfc))):
                continue

            RDD, NRP, RC, RB = get_action_terminals(
                req_state, time, num_req_process, remain_total_capacity[0], remain_total_bandwidth[0],
                len(active_requests), total_capacity, total_bandwidth
            )

            if (collect_situation and (time in range(start_t, end_t + 1))):
                act_data.append([
                    situation_act_id[0], RDD, NRP, RC, RB
                ])
            
            prio = action_rule(RDD, NRP, RC, RB)

            if (prio > threshold):
                process_requests.append(req_state)

        return process_requests

    # ==== Rule 2 ======================================================
    def order_request(process_requests):
        data = []

        if (collect_situation and (time in range(start_t, end_t + 1))):
            situation_ord_id[0] += 1

        for req_state in process_requests:
            BW, DC, RSFCL = get_order_terminals(
                req_state
            )

            #
            if (collect_situation and (time in range(start_t, end_t + 1))):
                ord_data.append([
                    situation_ord_id[0], BW, DC, RSFCL
                ])
            #

            prio = order_rule(BW, DC, RSFCL)

            data.append((req_state, prio))

        data.sort(key=lambda x: x[1], reverse=True)

        process_requests = [element[0] for element in data]

        return process_requests
        
    # ==== Rule 3 ======================================================
    def place_vnf(req_state):
        req = req_state["Request"]
        idx = req_state["Current_vnf"]
        vnf = req.sfc[idx]

        best_node = None
        best_prio = float("-inf")

        if (collect_situation and (time in range(start_t, end_t + 1)) and (time % 4 == 0)):
            situation_place_id[0] += 1

        for node in server_node:
            if (not node.can_host(vnf)):
                #
                # capacity = node.release()
                # remain_total_capacity[0] += capacity
                #
                continue

            NU, ND, NC, NN = get_placement_terminals(node, graph, vnf, time)

            #
            if (collect_situation and (time in range(start_t, end_t + 1)) and (time % 4 == 0)):
                place_data.append([
                    situation_place_id[0], NU, ND, NC, NN
                ])
            #

            # if (node.vnf_dict[vnf] != -1):
            #     continue

            prio = placement_rule(NU, ND, NC, NN)

            if (prio > best_prio):
                best_prio = prio
                best_node = node

        if (best_node is not None):

            if (best_node.vnf_dict[vnf] == -1):
                best_node.deploy(vnf, time)
            # remain_total_capacity[0] -= capacity
            req_state["Placement"].append((best_node, vnf))

        return best_node, vnf
    
    # ==== Rule 4 ======================================================
    def get_candidate_paths(graph, src, target, bandwidth, max_hop, k=50):
        paths = []

        def dfs(curr, visited, path):
            # if (len(paths) >= k):
            #     return
            if (len(path) > max_hop):
                return
            # if ((target in curr.vnf_dict) and (curr.vnf_dict[target] != -1) and (curr.can_host(target))):
            if (curr is target):
                paths.append(path.copy())
                return
            # neighbors = sorted(
            #     graph.neighbors(curr).items(),
            #     key=lambda x: x[1].delay
            # )
            for nxt, link in graph.neighbors(curr).items():
                if (not link.can_route(bandwidth)):
                    continue
                if (nxt not in visited):
                    visited.add(nxt)
                    path.append(link)
                    dfs(nxt, visited, path)
                    path.pop()
                    visited.remove(nxt)

        dfs(src, {src}, [])
        return paths
    
    # def is_path_feasible(path, bandwidth):
    #     for link in path:
    #         if (not link.can_route(bandwidth)):
    #             return False
    #     return True
        
    def route_vnf(req_state, best_node, vnf, max_hop=10):    # 15(nsf, cogent); 18(cogent); 8(conus)  
        # idx = req_state["Current_vnf"]
        # vnf = req_state["Request"].sfc[idx]
        bandwidth = req_state["Request"].bandwidth
        
        src = req_state["Current_location"]
        
        if (best_node is None):
            return False
        
        target = best_node

        # print(f"\n- Src: {src}; Target: {target}")

        paths = get_candidate_paths(graph, src, target, bandwidth, max_hop)
        # print(len(paths))
        # print(f"- Paths: {paths}\n")
        if (not paths):
            return False

        best_path = None
        best_prio = float("-inf")

        if (collect_situation and (time in range(start_t, end_t + 1)) and (time % 4 == 0)):
            situation_route_id[0] += 1

        for path in paths:
            # if (not is_path_feasible(path, bandwidth)):
            #     print("error")
            #     continue
            
            # node = path[-1].v if (path) else src
            PD, MLU, PL = get_routing_terminals(path)

            #
            if (collect_situation and (time in range(start_t, end_t + 1)) and (time % 4 == 0)):
                route_data.append([
                    situation_route_id[0], PD, MLU, PL
                ])
            #

            prio = routing_rule(PD, MLU, PL)

            if (prio > best_prio):
                best_prio = prio
                best_path = path

        # print(f"- Best_path -> {best_path}")
        if (best_path is None):
            return False

        neighbors = graph.neighbors
        for link in best_path:
            u = link.u
            v = link.v
            link_uv = neighbors(u).get(v)
            link_uv.route(bandwidth)

            link_vu = neighbors(v).get(u)
            link_vu.route(bandwidth)

            mlu[0] = max(mlu[0], link.get_link_utilization())
            remain_total_bandwidth[0] -= bandwidth
        
        if (best_path):
            dst_node = best_path[-1].v
            req_state["Process_time"] = compute_process_time(best_path)
        else:
            dst_node = src
            req_state["Process_time"] = dst_node.delay

        req_state["Path"].append(best_path)
        req_state["Current_location"] = dst_node

        capacity = dst_node.using(vnf)
        remain_total_capacity[0] -= capacity

        req_state["Cost"] += dst_node.get_total_cost(vnf)
        total_cost[0] += dst_node.get_total_cost(vnf)
        req_state["Process_time"] += max(dst_node.vnf_valid_time[vnf] - time, 0)

        req_state["Using"].append((dst_node, vnf))

        req_state["Current_vnf"] += 1

        #
        if (len(req_state["Path"]) > 1):
            release_sub_path(req_state, path_idx=-2, node_idxs=[-3])
        #

        return True
    # ===================================================================
    
    def reject(req_state):
        # release_resources(req_state)
        release_sub_path(req_state, path_idx=-1, node_idxs=[-2, -1])
        # release_node()

        active_requests.remove(req_state)
        rejected_requests.append(req_state)
        # print("\nReject:", req_state)
        # print(f"Reject {len(rejected_requests)} request!")

    def final_step(req_state):
        r = req_state["Request"]
        egress = req_state["Egress"]
        
        nodes[r.dst_node].deploy(egress, time)
    
        route_vnf(req_state, best_node=nodes[r.dst_node], vnf=egress, max_hop=15)   # 20(nsf, cogent); 22(cogent); 12(conus)

    def finish(req_state):
        # release_resources(req_state)
        release_sub_path(req_state, path_idx=-1, node_idxs=[-2, -1])
        # release_node()

        active_requests.remove(req_state)
        finished_requests.append(req_state)

    def release_sub_path(req_state, path_idx, node_idxs):
        # if len(using) == 1 then chỉ release node else then như cũ

        if (len(req_state["Using"]) == 1):
            node, vnf = req_state["Using"][0]

            capacity = node.not_using(vnf)
            remain_total_capacity[0] += capacity

        else:
            for node_idx in node_idxs:
                node, vnf = req_state["Using"][node_idx]
                
                capacity = node.not_using(vnf)
                remain_total_capacity[0] += capacity
            
            req = req_state["Request"]
            path = req_state["Path"][path_idx]

            for link in path:
                u = link.u
                v = link.v
                link_uv = graph.neighbors(u).get(v)
                link_uv.release(req.bandwidth)

                link_vu = graph.neighbors(v).get(u)
                link_vu.release(req.bandwidth)

                remain_total_bandwidth[0] += req.bandwidth

    # def release_node():
    #     for node in nodes.values():
    #     #     total_capacity = node.release()
    #     #     remain_total_capacity[0] += total_capacity
    #     # for node, _ in req_state["Using"]:
    #         node.release()
        
    # def release_resources(req_state):
    #     req = req_state["Request"]

    #     # release node capacity
    #     for node, vnf in req_state["Using"]:
    #         capacity = node.not_using(vnf)
    #         remain_total_capacity[0] += capacity

    #     # release link bandwidth
    #     neighbors = graph.neighbors
    #     for sub_path in req_state["Path"]:
    #         for link in sub_path:
    #             u = link.u
    #             v = link.v
    #             link_uv = neighbors(u).get(v)
    #             # if (link_uv):
    #             link_uv.release(req.bandwidth)

    #             link_vu = neighbors(v).get(u)
    #             # if (link_vu):
    #             link_vu.release(req.bandwidth)

    #             remain_total_bandwidth[0] += req.bandwidth

    #     for node, _ in req_state["Using"]:
    #     # for node in nodes.values():
    #         node.release()
    #         # remain_total_capacity[0] += capacity
            
    # Sửa thời gian xử lý node chỉ tính node sử dụng hoặc tính hết
    def compute_process_time(path):
        links_delay = 0
        # node_delay = 0

        for link in path:
            links_delay += link.delay
            # node_delay += link.v.delay

        node_delay = path[-1].v.delay
        process_time = node_delay + links_delay
        # print(f"\n- Process_time: {nodes_delay} + {links_delay} = {process_time}")

        return process_time

    # Main loop
    while ((len(finished_requests) + len(rejected_requests)) < len(requests)):
        # sleep(0.01)
        # print(f"Fin: {len(finished_requests)}, Rej: {len(rejected_requests)}, Req: {len(requests)}, Active: {len(active_requests)}")
        if (time in range(start_t, end_t + 1)):
            request_coming()

        process_requests = action_request()

        process_requests = order_request(process_requests)

        for req_state in process_requests:
            # print(f"\nReq: {req_state["Request"]}")
            best_node, vnf = place_vnf(req_state)
            # print(f"- Place -> {req_state["Placement"]}")
            route_vnf(req_state, best_node, vnf)
            # print(f"- Route -> {req_state["Path"]}")
    
        # Xử lý
        for req_state in active_requests[:]:
            if (req_state["Request"].due_date <= time):
                reject(req_state)
            elif (req_state["Process_time"] > 0):
                req_state["Process_time"] = max(req_state["Process_time"] - 0.05, 0)
                # print("\nAfter process:")
                # info_format(req_state)
            elif (req_state["Current_vnf"] == len(req_state["Request"].sfc)):
                # print("\nFinal_step:")
                final_step(req_state)
                # info_format(req_state)
            elif (req_state["Current_vnf"] > len(req_state["Request"].sfc)):
                req_state["Finish_time"] = time
                finish(req_state)
                # print("\nFinish:", req_state)

        # Giải phóng những node không dùng
        for node in nodes.values():
            node.release()
            # remain_total_capacity[0] += capacity
    
        time = round(time + 0.05, 2)

    # print(f"\n----\nFin: {len(finished_requests)}, Rej: {len(rejected_requests)}, Req: {len(requests)}, Active: {len(active_requests)}")

    if (display):
        print("After:")
        print("Total capacity:", total_capacity)
        print("Total bandwidth:", total_bandwidth)
        print("Remain total capacity:", remain_total_capacity)
        print("Remain total bandwidth:", remain_total_bandwidth)

        for node in nodes.values():
            for key, val in node.vnf_dict.items():
                if (val != -1):
                    print(node.name, key, val)


    if (display):
        print("\nFinished_Req:", len(finished_requests))
        print("Rejected_Req:", len(rejected_requests))
        print("Requests:", len(requests))
        print("Active_Req:", len(active_requests))
        
        print("\nTotal_cost:", total_cost[0])
        print("Num rejected_requests:", len(rejected_requests))
        print("MLU:", mlu[0])

    if (collect_situation):
        act_columns = ["situation_id", "RDD", "NRP", "RC", "RB"]
        pd.DataFrame(act_data, columns=act_columns).to_csv("situations\\act_situations.csv", index=False)

        ord_columns = ["situation_id", "BW", "DC", "RSFCL"]
        pd.DataFrame(ord_data, columns=ord_columns).to_csv("situations\\ord_situations.csv", index=False)

        place_columns = ["situation_id", "NU", "ND", "NC", "NN"]
        pd.DataFrame(place_data, columns=place_columns).to_csv("situations\\place_situations.csv", index=False)

        route_columns = ["situation_id", "PD", "MLU", "PL"]
        pd.DataFrame(route_data, columns=route_columns).to_csv("situations\\route_situations.csv", index=False)

        print(f"\nCollect situation successful!")

    return total_cost[0], len(rejected_requests)

# Reference
def ref_action_rule(RDD, NRP, RC, RB):
    return RDD

def ref_order_rule(BW, DC, RSFCL):
    return -DC
    # return -RSFCL
    # return -(BW * RSFCL * DC)

def ref_placement_rule(NU, ND, NC, NN):
    # return -(NU * ND * NC * (1 / NN))
    return -NC
    # return -ND

def ref_routing_rule(PD, MLU, PL):
    return -PD
    # return -(PD * MLU * PL)

# First Fit
def first_fit_action_rule(RDD, NRP, RC, RB):
    return 1

def first_fit_order_rule(BW, DC, RSFCL):
    return 1

def first_fit_placement_rule(NU, ND, NC, NN):
    return 1

def first_fit_routing_rule(PD, MLU, PL):
    return 1

# Greedy
def greedy_action_rule(RDD, NRP, RC, RB):
    return RDD

def greedy_order_rule(BW, DC, RSFCL):
    return -DC  # v2
    # return -RSFCL # v1

def greedy_placement_rule(NU, ND, NC, NN):
    return -NC
    # return -ND

def greedy_routing_rule(PD, MLU, PL):
    return -PD

# Random
def random_action_rule(RDD, NRP, RC, RB):
    return random.uniform(-1, 1)

def random_order_rule(BW, DC, RSFCL):
    return random.uniform(-1, 1)

def random_placement_rule(NU, ND, NC, NN):
    return random.uniform(-1, 1)

def random_routing_rule(PD, MLU, PL):
    return random.uniform(-1, 1)

# # Combine
# def combine_action_rule(RDD, NRP, RC, RB):
#     return random.uniform(-1, 1)

# def combine_order_rule(BW, DC, RSFCL):
#     return 1

# def combine_placement_rule(NU, ND, NC, NN):
#     return -ND

# def combine_routing_rule(PD, MLU, PL):
#     return -PD

def info_format(req_state):
    print(f"""\n- Request: {req_state["Request"]}
    + Current_vnf: {req_state["Current_vnf"]}
    + Current_location: {req_state["Current_location"]}
    + Placement: {req_state["Placement"]}
    + Path: {req_state["Path"]}
    + Finish_time: {req_state["Finish_time"]}
    + Cost: {req_state["Cost"]}
    + Process_time: {req_state["Process_time"]}
    """)

# start = time.time()

# graph, nodes, requests, total_capacity, total_bandwidth = read_data(
#     "input_25\\cogent_uniform_normal_s2.json"
# )

# T = max([req.arrival_time for req in requests])
# start_t = 0
# end_t = int(T * 0.5)

# start_t = end_t + 1
# end_t = T

# requests = [req for req in requests if ((req.arrival_time >= start_t) and (req.arrival_time <= end_t))]

# print("Ref")
# evaluate(graph, nodes, requests,
#         start_t, end_t,
#         total_capacity, total_bandwidth,
#         action_rule=ref_action_rule, order_rule=ref_order_rule, placement_rule=ref_placement_rule, routing_rule=ref_routing_rule, 
#         display=True, collect_situation=False)

# print("\nGreedy")
# evaluate(graph, nodes, requests,
#         start_t, end_t,
#         total_capacity, total_bandwidth,
#         action_rule=greedy_action_rule, order_rule=greedy_order_rule, placement_rule=greedy_placement_rule, routing_rule=greedy_routing_rule, 
#         display=True, collect_situation=False)

# print("\nCombine")
# evaluate(graph, nodes, requests,
#         start_t, end_t,
#         total_capacity, total_bandwidth,
#         action_rule=combine_action_rule, order_rule=combine_order_rule, placement_rule=combine_placement_rule, routing_rule=combine_routing_rule, 
#         display=True, collect_situation=False)

# print("\nRandom")
# evaluate(graph, nodes, requests,
#         start_t, end_t,
#         total_capacity, total_bandwidth,
#         action_rule=random_action_rule, order_rule=random_order_rule, placement_rule=random_placement_rule, routing_rule=random_routing_rule, 
#         display=True, collect_situation=False)
 
# end = time.time()
# print(f"Time: {round(end - start, 4)} s")
