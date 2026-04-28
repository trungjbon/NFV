from deap import base, creator, gp, tools
import operator
import random
from utils import *
from functools import partial
import copy
import os
import time

# =========================
# Load data
# =========================
# graph_raw, nodes_raw, requests_raw, server_cnt_raw = read_data(
#     "test_nsf_centers_easy_s1.json"
# )

# =========================
# Protected operators
# =========================

def div_op(left, right):
    if (right != 0):
        return left / right
    return 1

def max_op(left, right):
    return max(left, right)

def min_op(left, right):
    return min(left, right)

# =========================
# Primitive sets
# =========================

# ---- Action rule ----
pset_act = gp.PrimitiveSet("ACT", 4)
pset_act.addPrimitive(operator.add, 2)
pset_act.addPrimitive(operator.sub, 2)
pset_act.addPrimitive(operator.mul, 2)
pset_act.addPrimitive(div_op, 2)
pset_act.addPrimitive(max_op, 2)
pset_act.addPrimitive(min_op, 2)
pset_act.renameArguments(       
    ARG0="RDD",
    ARG1="NRP", 
    ARG2="RC",
    ARG3="RB"
)
pset_act.addEphemeralConstant("CONST", partial(random.uniform, 0, 1.0))

# ---- Order rule ----
pset_ord = gp.PrimitiveSet("Ord", 3)
pset_ord.addPrimitive(operator.add, 2)
pset_ord.addPrimitive(operator.sub, 2)
pset_ord.addPrimitive(operator.mul, 2)
pset_ord.addPrimitive(div_op, 2)
pset_ord.addPrimitive(max_op, 2)
pset_ord.addPrimitive(min_op, 2)
pset_ord.renameArguments(
    ARG0="BW",  
    ARG1="DC",      
    ARG2="RSFCL",      
)
pset_ord.addEphemeralConstant("CONST", partial(random.uniform, 0, 1.0))

# ---- Placement rule ----
pset_place = gp.PrimitiveSet("PLACE", 4)
pset_place.addPrimitive(operator.add, 2)
pset_place.addPrimitive(operator.sub, 2)
pset_place.addPrimitive(operator.mul, 2)
pset_place.addPrimitive(div_op, 2)
pset_place.addPrimitive(max_op, 2)
pset_place.addPrimitive(min_op, 2)
pset_place.renameArguments(
    ARG0="NU",
    ARG1="ND",      # Node delay
    ARG2="NC",      # Node cost
    ARG3="NN"       # Number of neighboring nodes
)
pset_place.addEphemeralConstant("CONST", partial(random.uniform, 0, 1.0))

# ---- Routing rule ----
pset_route = gp.PrimitiveSet("ROUTE", 3)
pset_route.addPrimitive(operator.add, 2)
pset_route.addPrimitive(operator.sub, 2)
pset_route.addPrimitive(operator.mul, 2)
pset_route.addPrimitive(div_op, 2)
pset_route.addPrimitive(max_op, 2)
pset_route.addPrimitive(min_op, 2)
pset_route.renameArguments(
    ARG0="PD",      # Path delay
    ARG1="MLU",     # Maximum Link utilization
    ARG2="PL"       # Path length
)
pset_route.addEphemeralConstant("CONST", partial(random.uniform, 0, 1.0))

# =========================
# DEAP setup
# =========================
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Expression generators
toolbox.register("expr_act", gp.genHalfAndHalf, pset=pset_act, min_=2, max_=6) # 8
toolbox.register("expr_ord", gp.genHalfAndHalf, pset=pset_ord, min_=2, max_=6)
toolbox.register("expr_place", gp.genHalfAndHalf, pset=pset_place, min_=2, max_=6)
toolbox.register("expr_route", gp.genHalfAndHalf, pset=pset_route, min_=2, max_=6)

# Tree initializers
toolbox.register("tree_act", tools.initIterate, gp.PrimitiveTree, toolbox.expr_act)
toolbox.register("tree_ord", tools.initIterate, gp.PrimitiveTree, toolbox.expr_ord)
toolbox.register("tree_place", tools.initIterate, gp.PrimitiveTree, toolbox.expr_place)
toolbox.register("tree_route", tools.initIterate, gp.PrimitiveTree, toolbox.expr_route)

# Individual = 4 trees
def init_individual():
    return creator.Individual([
        toolbox.tree_act(),
        toolbox.tree_ord(),
        toolbox.tree_place(),
        toolbox.tree_route()
    ])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compile
toolbox.register("compile_act", gp.compile, pset=pset_act)
toolbox.register("compile_ord", gp.compile, pset=pset_ord)
toolbox.register("compile_place", gp.compile, pset=pset_place)
toolbox.register("compile_route", gp.compile, pset=pset_route)

# =========================
# Evaluation
# =========================
def evaluate_individual(ind, data_file, start_t=0, end_t=0, display=False, penalty_cost=0, penalty_reject=0):
    graph, nodes, requests, total_capacity, total_bandwidth = read_data(
        data_file
    )

    requests = [req for req in requests if ((req.arrival_time >= start_t) and (req.arrival_time <= end_t))]

    act_rule = toolbox.compile_act(ind[0])
    ord_rule = toolbox.compile_ord(ind[1])
    place_rule = toolbox.compile_place(ind[2])
    route_rule = toolbox.compile_route(ind[3])

    total_cost, num_rejected_requests = evaluate(
        graph=graph, nodes=nodes, requests=requests, 
        start_t=start_t, end_t=end_t,
        total_capacity=total_capacity, total_bandwidth=total_bandwidth,
        action_rule=act_rule,
        order_rule=ord_rule,
        placement_rule=place_rule,
        routing_rule=route_rule,
        display=display
    )
 
    reject_ratio = num_rejected_requests / len(requests)
    if (reject_ratio > 0.3):
        penalty_cost = (reject_ratio - 0.3) * penalty_cost * 2
        penalty_reject = (reject_ratio - 0.3) * penalty_reject

        total_cost += penalty_cost
        num_rejected_requests += penalty_reject

    return (total_cost, num_rejected_requests)


# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA2)

# =========================
# Genetic operators
# =========================
MAX_HEIGHT = 8

# ---- Crossover ----
cx = gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT)(gp.cxOnePoint)

def mate_individual(ind1, ind2):
    idx = random.randint(0, 3)
    ind1[idx], ind2[idx] = cx(ind1[idx], ind2[idx])

    # ind1[0], ind2[0] = cx(ind1[0], ind2[0])
    # ind1[1], ind2[1] = cx(ind1[1], ind2[1])
    # ind1[2], ind2[2] = cx(ind1[2], ind2[2])
    # ind1[3], ind2[3] = cx(ind1[3], ind2[3])

    return ind1, ind2

toolbox.register("mate", mate_individual)

# ---- Mutation ----
mut = gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT)(gp.mutUniform)

def mutate_individual(ind):
    idx = random.randint(0, 3)
    if (idx == 0):
        ind[0], = mut(ind[0], expr=toolbox.expr_act, pset=pset_act)
    elif (idx == 1):
        ind[1], = mut(ind[1], expr=toolbox.expr_ord, pset=pset_ord)
    elif (idx == 2):
        ind[2], = mut(ind[2], expr=toolbox.expr_place, pset=pset_place)
    else:
        ind[3], = mut(ind[3], expr=toolbox.expr_route, pset=pset_route)

    return (ind,)

toolbox.register("mutate", mutate_individual)

######################################################
import numpy as np
import pandas as pd
from utils import ref_action_rule, ref_order_rule, ref_placement_rule, ref_routing_rule
from scipy.stats import rankdata

def read_situations(path):
    df = pd.read_csv(path)
    selected_situations = np.random.choice(
        df["situation_id"].unique(), size=20, replace=False
    )

    return df[df["situation_id"].isin(selected_situations)]

def preprocess_situations(df, ref_rule, type):
    situations = {}

    for s_id, df_s in df.groupby("situation_id"):
        if (type == "act"):
            X = df_s[["RDD", "NRP", "RC", "RB"]].values
        elif (type == "ord"):
            X = df_s[["BW", "DC", "RSFCL"]].values
        elif (type == "place"):
            X = df_s[["NU", "ND", "NC", "NN"]].values
        elif (type == "route"):
            X = df_s[["PD", "MLU", "PL"]].values
        else:
            return None

        ref_scores = np.array([
            ref_rule(*row) for row in X
        ])

        ref_rank = rankdata(-ref_scores, method="min")

        situations[s_id] = {
            "X": X,
            "ref_rank": ref_rank
        }

    return situations

def semantic_vector(ind):
    act_rule = toolbox.compile_act(ind[0])
    ord_rule = toolbox.compile_ord(ind[1])
    place_rule = toolbox.compile_place(ind[2])
    route_rule = toolbox.compile_route(ind[3])

    vec = []

    for s in act_situations.values():
        scores = np.array([act_rule(*x) for x in s["X"]])
        best_idx = np.argmax(scores)
        vec.append(s["ref_rank"][best_idx])

    for s in ord_situations.values():
        scores = np.array([ord_rule(*x) for x in s["X"]])
        best_idx = np.argmax(scores)
        vec.append(s["ref_rank"][best_idx])

    for s in place_situations.values():
        scores = np.array([place_rule(*x) for x in s["X"]])
        best_idx = np.argmax(scores)
        vec.append(s["ref_rank"][best_idx])

    for s in route_situations.values():
        scores = np.array([route_rule(*x) for x in s["X"]])
        best_idx = np.argmax(scores)
        vec.append(s["ref_rank"][best_idx])

    return np.array(vec)

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def surrogate_fitness(ind, archive, k=5):
#     archive = [
#       (semantic_vector, true_fitness),
#       ...
#     ]
    v = semantic_vector(ind)

    # _, best_fit = min(
    #     archive,
    #     key=lambda x: euclidean(v, x[0])
    # )

    # return best_fit

    neighbors = sorted(
        archive,
        key=lambda x: euclidean(v, x[0])
    )[:k]

    weights = []
    fits = []

    for vec, fit in neighbors:
        d = euclidean(v, vec) + 1e-6
        weights.append(1 / d)   # Càng giống -> càng ảnh hưởng mạnh
        fits.append(fit)

    weights = np.array(weights)
    fits = np.array(fits)

    weighted_fit = np.sum(weights[:, None] * fits, axis=0) / np.sum(weights)

    return tuple(weighted_fit)   # inverse-distance weighted kNN

from sklearn.cluster import KMeans

def cluster_population(pop, n_clusters):
    vectors = np.array([semantic_vector(ind) for ind in pop])

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = kmeans.fit_predict(vectors)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append((pop[i], vectors[i]))

    return clusters

def individual_size(ind):
    return sum(len(tree) for tree in ind)   # tổng node của 4 cây

def select_samples(clusters):
    samples = []

    for cluster in clusters.values():
        rep = min(cluster, key=lambda x: individual_size(x[0]))
        samples.append(rep)

    return samples

# =========================
# GP main loop
# =========================

def genetic_programming(pop_size=100, generation=30, cx_prob=0.8, mut_prob=0.2):
    # Khởi tạo Archive tốt nhất mọi thời đại
    best_archive = tools.ParetoFront()

    # front = None
    
    pop = toolbox.population(n=pop_size)
    
    for gen in range(generation):
        print(f"\nGeneration {gen}")
        # 0. Reset fitness
        for ind in pop:
            del ind.fitness.values

        # 1. Select Individuals as Sample
        n_clusters = pop_size // 5
        clusters = cluster_population(pop, n_clusters=n_clusters)
        samples = select_samples(clusters)

        print("Done Select Individuals as Sample")

        archive = []
        # 2. Evaluate Specific Individuals (real fitness)
        for ind, vec in samples:
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit

            # CẬP NHẬT
            best_archive.update([ind])

        # 3. Build Surrogate (implicit qua archive)
            archive.append((vec, fit))
            print(f"\tDone evaluate {len(archive)} ind")

        print("Done Build Surrogate")
        
        # 4. Estimate Remaining Individuals
        for ind in pop:
            if (not ind.fitness.valid):
                fit = surrogate_fitness(ind, archive)
                ind.fitness.values = fit

        print("Done Estimate Remaining Individuals")

        # 5. Parent Selection
        pop = toolbox.select(pop, len(pop))
        parents = tools.selTournamentDCD(pop, k=pop_size)
        offspring = [toolbox.clone(ind) for ind in parents]

        print("Done Parent Selection")

        # 6. Breed New Population
        # Crossover
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if (random.random() < cx_prob):
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        # Mutation
        for ind in offspring:
            if (random.random() < mut_prob):
                toolbox.mutate(ind)
                del ind.fitness.values

        print("Done Breed New Population")

        # 7. Evaluate offspring (surrogate)
        clusters = cluster_population(offspring, n_clusters=n_clusters)
        samples = select_samples(clusters)

        for ind, vec in samples:
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit

            # CẬP NHẬT
            best_archive.update([ind])

            archive.append((vec, fit))
            print(f"\tDone evaluate {len(archive)} ind")

        for ind in offspring:
            if (not ind.fitness.valid):
                fit = surrogate_fitness(ind, archive)
                ind.fitness.values = fit

        print("Done Evaluate offspring")

        # 8. Replace population
        pop = toolbox.select(pop + offspring, k=pop_size)

        print("Done Replace population")

        # 9. Logging
        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        print(f"Pareto front size: {len(front)}")
        f1 = [ind.fitness.values[0] for ind in front]
        f2 = [ind.fitness.values[1] for ind in front]
        print(f"Min cost: {min(f1)}")
        print(f"Min reject: {min(f2)}")
    
        f2_archive = [ind.fitness.values[1] for ind in best_archive]
        print(f"Global Min reject (Archive): {min(f2_archive)}")

    # return front
    return best_archive

# =========================
# Run
# =========================

data_folder = "input_25"
# data_folder = "data_1_9"
result_file = "result02\\end02_r42\\conus_urban_easy_s3.txt"
result_file_csv = "result02\\end02_r42\\conus_urban_easy_s3.csv"

# data_files = [f for f in os.listdir(data_folder) if (f.startswith("nsf_uniform_easy") or f.startswith("nsf_uniform_normal"))]

data_files = [f for f in os.listdir(data_folder) if (f.startswith("conus_urban_easy_s3"))]

import csv

with open(result_file_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Objective: (Cost, Num rejected reqs)"])
    writer.writerow(["DATASET", "GP reject", "",
                     "Heuristic 1 (Greedy)", "", "Heuristic 2 (First Fit)", "", "Heuristic 3 (Random)", ""])  # header
    writer.writerow(["", "Fitness", "Percentage rejected", 
                     "Fitness", "Percentage rejected", "Fitness", "Percentage rejected", "Fitness", "Percentage rejected"])

    with open(result_file, "w") as f:
        for file in data_files:
            data_path = os.path.join(data_folder, file)
            set_seed(SEED)
            print(f"===== Running {file} =====")
            f.write(f"- DATASET: {file}\n")
            f.write(f"- Objective: (Cost, Num rejected reqs)\n")

            start = time.time()
            # ===== Load data =====
            graph, nodes, requests, total_capacity, total_bandwidth = read_data(data_path)
            f.write(f"- Total requests: {len(requests)}\n\n")
            T = max([req.arrival_time for req in requests])

            # Train
            start_t = 0
            end_t = int(T * 0.5)
            f.write(f"\t- Train: start_t = {start_t}, end_t = {end_t}\n")

            reqs = [req for req in requests if ((req.arrival_time >= start_t) and (req.arrival_time <= end_t))]
            f.write(f"\t\t+ Total requests: {len(reqs)}\n")

            # Collect situation
            total_cost, num_rejected_requests = evaluate(
                graph, nodes, reqs,
                start_t, end_t,
                total_capacity, total_bandwidth,
                action_rule=ref_action_rule, order_rule=ref_order_rule, placement_rule=ref_placement_rule, routing_rule=ref_routing_rule, 
                display=True, collect_situation=True)
            
            df_act = read_situations("situations\\act_situations.csv")
            df_ord = read_situations("situations\\ord_situations.csv")
            df_place = read_situations("situations\\place_situations.csv")
            df_route = read_situations("situations\\route_situations.csv")

            act_situations = preprocess_situations(df_act, ref_action_rule, type="act")
            ord_situations = preprocess_situations(df_ord, ref_order_rule, type="ord")
            place_situations = preprocess_situations(df_place, ref_placement_rule, type="place")
            route_situations = preprocess_situations(df_route, ref_routing_rule, type="route")

            if ("evaluate" in toolbox.__dict__):
                toolbox.unregister("evaluate")
            toolbox.register("evaluate", partial(
                evaluate_individual, data_file=data_path, start_t=start_t, end_t=end_t, penalty_cost=total_cost, penalty_reject=num_rejected_requests
            ))

            pareto_archive = genetic_programming()
            # f.write(f"\n\t+ Pareto front size: {len(pareto_front)}")
            # for i, ind in enumerate(pareto_archive):
            #     print(f"Sol {i}: {ind.fitness.values}")
            print(f"Len pareto: {len(pareto_archive)}")
            
            # k = min(len(pareto_archive), 20)
            # best_inds = sorted(pareto_archive, key=lambda ind: ind.fitness.values[1])[:k]
            
            # best_ind_reject = min(pareto_archive, key=lambda ind: ind.fitness.values[1])

            # f.write(f"\t\t+ Selected reject solution:\n")
            # f.write(f"\t\t\t> ACT rule: {best_ind_reject[0]}\n")
            # f.write(f"\t\t\t> ORD rule: {best_ind_reject[1]}\n")
            # f.write(f"\t\t\t> PLACE rule: {best_ind_reject[2]}\n")
            # f.write(f"\t\t\t> ROUTE rule: {best_ind_reject[3]}\n")
            # f.write(f"\t\t\t> Finess: {best_ind_reject.fitness.values}\n\n")

            # Test
            start_t = end_t + 1
            end_t = T
            f.write(f"\t- Test: start_t = {start_t}, end_t = {end_t}\n")

            reqs = [req for req in requests if ((req.arrival_time >= start_t) and (req.arrival_time <= end_t))]
            f.write(f"\t\t+ Total requests: {len(reqs)}\n")

            f.write(f"\t\t+ Len pareto: {len(pareto_archive)}\n")

            #
            best_ind = None
            best_cost_test = float("inf")
            best_reject_test = float("inf")

            for i, ind in enumerate(pareto_archive):
                print(f"\nEvaluate ind {i} in test")
                f.write(f"\t\t\t> Evaluate ind {i} in test\n")

                fitness_test = evaluate_individual(ind, data_file=data_path, start_t=start_t, end_t=end_t)
                print(f"Fitness: {ind.fitness.values}")
                print(f"Fitness test: {fitness_test}")
                f.write(f"\t\t\t> Fitness: {ind.fitness.values}\n")
                f.write(f"\t\t\t> Fitness test: {fitness_test}\n\n")

                if ((fitness_test[1] < best_reject_test) or 
                    ((fitness_test[1] == best_reject_test) and (fitness_test[0] < best_cost_test))):
                    best_cost_test = fitness_test[0]
                    best_reject_test = fitness_test[1]
                    best_ind = ind

            f.write(f"\t\t+ Selected reject solution:\n")
            f.write(f"\t\t\t> ACT rule: {best_ind[0]}\n")
            f.write(f"\t\t\t> ORD rule: {best_ind[1]}\n")
            f.write(f"\t\t\t> PLACE rule: {best_ind[2]}\n")
            f.write(f"\t\t\t> ROUTE rule: {best_ind[3]}\n")
            f.write(f"\t\t\t> Finess: {best_ind.fitness.values}\n\n")
            #

            # fitness_reject = evaluate_individual(best_ind_reject, data_file=data_path, start_t=start_t, end_t=end_t, display=True)
            fitness_reject = evaluate_individual(best_ind, data_file=data_path, start_t=start_t, end_t=end_t, display=True)
            f.write(f"\t\t+ Fitness reject: {fitness_reject}\n")
            percentage_rejected_by_gp_reject = round(fitness_reject[1] / len(reqs), 2)
            f.write(f"\t\t+ Percentage of rejected requests: {percentage_rejected_by_gp_reject}%\n\n")

            # Heuristic 1
            f.write(f"\t- Heuristic 1 (Greedy): start_t = {start_t}, end_t = {end_t}\n")
            f.write(f"\t\t+ Total requests: {len(reqs)}\n")
            fitness_heuristic1 = evaluate(graph=graph, nodes=nodes, requests=reqs, 
                                        start_t=start_t, end_t=end_t,
                                        total_capacity=total_capacity, total_bandwidth=total_bandwidth,
                                        action_rule=greedy_action_rule, 
                                        order_rule=greedy_order_rule, 
                                        placement_rule=greedy_placement_rule,
                                        routing_rule=greedy_routing_rule,
                                        display=True)
            f.write(f"\t\t+ Fitness: {fitness_heuristic1}\n")
            percentage_rejected_by_heuristic1 = round(fitness_heuristic1[1] / len(reqs), 2)
            f.write(f"\t\t+ Percentage of rejected requests: {percentage_rejected_by_heuristic1}%\n\n")

            # Heuristic 2
            f.write(f"\t- Heuristic 2 (First fit): start_t = {start_t}, end_t = {end_t}\n")
            f.write(f"\t\t+ Total requests: {len(reqs)}\n")
            fitness_heuristic2 = evaluate(graph=graph, nodes=nodes, requests=reqs, 
                                        start_t=start_t, end_t=end_t,
                                        total_capacity=total_capacity, total_bandwidth=total_bandwidth,
                                        action_rule=first_fit_action_rule, 
                                        order_rule=first_fit_order_rule, 
                                        placement_rule=first_fit_placement_rule,
                                        routing_rule=first_fit_routing_rule)
            f.write(f"\t\t+ Fitness: {fitness_heuristic2}\n")
            percentage_rejected_by_heuristic2 = round(fitness_heuristic2[1] / len(reqs), 2)
            f.write(f"\t\t+ Percentage of rejected requests: {percentage_rejected_by_heuristic2}%\n\n")
            
            # Heuristic 3
            f.write(f"\t- Heuristic 3 (Random): start_t = {start_t}, end_t = {end_t}\n")
            f.write(f"\t\t+ Total requests: {len(reqs)}\n")
            fitness_heuristic3 = evaluate(graph=graph, nodes=nodes, requests=reqs, 
                                        start_t=start_t, end_t=end_t,
                                        total_capacity=total_capacity, total_bandwidth=total_bandwidth,
                                        action_rule=random_action_rule, 
                                        order_rule=random_order_rule, 
                                        placement_rule=random_placement_rule,
                                        routing_rule=random_routing_rule)
            f.write(f"\t\t+ Fitness: {fitness_heuristic3}\n")
            percentage_rejected_by_heuristic3 = round(fitness_heuristic3[1] / len(reqs), 2)
            f.write(f"\t\t+ Percentage of rejected requests: {percentage_rejected_by_heuristic3}%\n\n")

            end = time.time()
            f.write(f"- Time: {end - start}s\n\n")
 
            writer.writerow([file,
                             fitness_reject, percentage_rejected_by_gp_reject,
                             fitness_heuristic1, percentage_rejected_by_heuristic1,
                             fitness_heuristic2, percentage_rejected_by_heuristic2,
                             fitness_heuristic3, percentage_rejected_by_heuristic3])
            f.write(40 * "-")
            f.write("\n\n")
