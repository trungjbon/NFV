class Node:
    def __init__(self, name, type, delay, capacity, cost):
        self.name = name
        self.type = type
        self.delay = delay
        self.capacity = capacity
        self.used_capacity = {
            "cpu": 0.0,
            "ram": 0.0,
            "mem": 0.0
        }
        self.cost = cost
        self.vnf_dict = {}  # {vnf: num of req using} 
        self.vnf_valid_time = {}

    def can_host(self, vnf):
        for key in self.used_capacity.keys():
            if (self.used_capacity[key] + vnf.demand_capacity[key] > self.capacity[key]):
                return False
    
        return True
    
    def using(self, vnf):
        self.vnf_dict[vnf] += 1

    def not_using(self, vnf):
        self.vnf_dict[vnf] -= 1
    
    def deploy(self, vnf, time):
        total_capacity = 0

        self.vnf_valid_time[vnf] = (vnf.boot_time[self.name] + time) if (vnf.boot_time is not None) else time
        self.vnf_dict[vnf] = 0
        for key in self.used_capacity.keys():
            self.used_capacity[key] += vnf.demand_capacity[key]

        total_capacity += vnf.get_total_demand_capacity()

        return total_capacity


    # def update_valid_time(self, vnf, time):
    #     self.vnf_valid_time[vnf] = time
        
    def release(self):
        total_capacity = 0

        for vnf in self.vnf_dict.keys():
            if (self.vnf_dict[vnf] == 0):
                for key in self.used_capacity.keys():
                    self.used_capacity[key] -= vnf.demand_capacity[key]
                total_capacity += vnf.get_total_demand_capacity()
                self.vnf_dict[vnf] = -1
                self.vnf_valid_time[vnf] = float("inf")
        
        return total_capacity

    def get_total_used_capacity(self):
        return sum(self.used_capacity.values())
    
    def get_total_capacity(self):
        return sum(self.capacity.values())
    
    def get_total_cost(self, vnf):
        total_cost = 0

        for name in self.cost.keys():
            total_cost += self.cost[name] * vnf.demand_capacity[name]

        return total_cost

    def __repr__(self):
        # return f"Node({self.name}, cap={self.used_capacity}/{self.capacity})"
        return f"Node({self.name}, type={self.type}, delay={self.delay}, cap={self.capacity}, cost={self.cost})"
    

class Link:
    def __init__(self, u, v, bandwidth, delay):
        self.u = u
        self.v = v
        self.bandwidth = bandwidth
        self.used_bandwidth = 0
        self.delay = delay

    def can_route(self, bandwidth):
        return self.used_bandwidth + bandwidth <= self.bandwidth
    
    def route(self, bandwidth):
        self.used_bandwidth += bandwidth
    
    def release(self, bandwidth):
        self.used_bandwidth -= bandwidth

    def get_link_utilization(self):
        return self.used_bandwidth / self.bandwidth

    def __repr__(self):
        return f"Link(u={self.u}, v={self.v}, delay={self.delay}, bw={self.used_bandwidth}/{self.bandwidth})"


class Graph:
    def __init__(self, directed):
        self.adj = {}
        self.num_links = 0
        self.directed = directed

    def add_node(self, node):
        if (node not in self.adj):
            self.adj[node] = {}

    def add_edge(self, u, v, bandwidth, delay):
        self.add_node(u)
        self.add_node(v)

        link = Link(u, v, bandwidth, delay)
        self.adj[u].update({v: link})

        if (not self.directed):
            link = Link(v, u, bandwidth, delay)
            self.adj[v].update({u: link})

        self.num_links += 1

    def neighbors(self, node):
        return self.adj.get(node, {})

    def __repr__(self):
        return "\n".join(f"{u} -> {self.adj[u]}" for u in self.adj)
    
class VNF:
    def __init__(self, name, cpu_demand=0, ram_demand=0, mem_demand=0, boot_time=None):
        self.name = name
        self.demand_capacity = {
            "cpu": cpu_demand,
            "ram": ram_demand,
            "mem": mem_demand
        }
        self.boot_time = boot_time

    def get_total_demand_capacity(self):
        return sum(self.demand_capacity.values())

    def __repr__(self):
        return f"VNF({self.name}, demand={self.demand_capacity}, boot_time={self.boot_time})"

class Request:
    def __init__(self, name, arrival_time, src_node, dst_node, sfc, bandwidth, due_date):
        self.name = name
        self.arrival_time = arrival_time
        self.src_node = src_node
        self.dst_node = dst_node
        self.sfc = sfc
        self.bandwidth = bandwidth
        self.due_date = due_date
        
    def __repr__(self):
        return f"Request({self.name}, bw={self.bandwidth}, SFC={self.sfc}, due_date={self.due_date})"
    

########################################################################

# node_ingress = Node(id="ingress", delay=0, capacity=0)
# node_a = Node(id="a", delay=50, capacity=8)
# node_b = Node(id="b", delay=40, capacity=11)
# node_c = Node(id="c", delay=80, capacity=15)
# node_d = Node(id="d", delay=60, capacity=12)
# node_egress = Node(id="egress", delay=0, capacity=0)

# network = Graph()
# network.add_edge(node_ingress, node_a, link_delay=0)
# network.add_edge(node_a, node_b, link_delay=15)
# network.add_edge(node_a, node_c, link_delay=10)
# network.add_edge(node_b, node_d, link_delay=15)
# network.add_edge(node_c, node_d, link_delay=25)
# network.add_edge(node_d, node_egress, link_delay=0)

# print(network)

# vnf1 = VNF(id="f1", capacity=8)
# vnf2 = VNF(id="f2", capacity=6)
# vnf3 = VNF(id="f3", capacity=14)
# vnf4 = VNF(id="f4", capacity=5)

# request1 = Request(id="r1", data_rate=150, vnfs=[vnf1, vnf3])
# request2 = Request(id="r2", data_rate=200, vnfs=[vnf1, vnf2])
# request3 = Request(id="r1", data_rate=300, vnfs=[vnf2, vnf3, vnf4])

# def route_between(u, v, rate):
#     visited = set()
#     queue = [(u, [])]

#     while (queue):
#         curr, path = queue.pop(0)
#         if (curr == v):
#             return path
        
#         for next, link in graph.neighbors(curr):
#             if ((next not in visited) and link.can_route(rate)):
#                 visited.add(next)
#                 queue.append((next, path + [link]))

#     return None


# def select_path(paths, routing_rule, data_rate):
#         best_path = None
#         best_prio = None

#         for path in paths:
#             if (not is_path_feasible(path, data_rate)):
#                 continue

#             BWU, PL, PD, DR = get_routing_terminals(path, data_rate)
#             prio = routing_rule(BWU, PL, PD, DR)

#             if ((best_prio is None) or (prio > best_prio)):
#                 best_prio = prio
#                 best_path = path

#         return best_path