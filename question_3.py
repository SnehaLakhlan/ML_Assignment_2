import json
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from data_setup import X_train, y_train, X_test, y_test, test_masked, seed
from question_1 import bagged_trees, disagreement_and_margin, bagging_votes
from question_2 import SAMMEAdaboost

num_repeats = 30


def make_resampled_indexes(y, num_repeats, rng):
    classes = np.unique(y)
    all_indexes = []
    for _ in range(num_repeats):
        resampled_indexes = []
        for c in classes:
            class_indexes = np.where(y == c)[0]
            resampled_indexes.extend(rng.choice(class_indexes, size=len(class_indexes), replace=True))
        all_indexes.append(resampled_indexes)
    return np.array(all_indexes)


def tree_proba(model, X):
    return model.predict_proba(X)  # every resampled set keeps all 40 identities


def bagging_proba(model, X):
    return bagging_votes(model, X) / model.n_trees


def boosting_proba(model, X):
    return model.confidence(X)


def bias_variance(prob_tensor, y):
    p_bar = prob_tensor.mean(axis=0)
    e = np.zeros_like(p_bar)
    e[np.arange(len(y)), y] = 1
    bias_squared = np.mean(np.sum((p_bar - e) ** 2, axis=1))
    variance = np.mean(np.sum((prob_tensor - p_bar) ** 2, axis=2))
    return bias_squared, variance


def run_refits(make_model, proba_fn, resampled_sets):
    # every model is refitted on the same 30 resampled training sets
    prob_list, acc_list, disagree_list = [], [], []
    for r, indexes in enumerate(resampled_sets):
        model = make_model(r)
        model.fit(X_train[indexes], y_train[indexes])
        acc_list.append(accuracy_score(y_test, model.predict(X_test)))
        prob_list.append(proba_fn(model, X_test))
        if isinstance(model, bagged_trees):
            disagree_list.append(disagreement_and_margin(model, X_test)[0])
    return np.array(prob_list), np.array(acc_list), np.array(disagree_list)


def leaf_occupancy(max_depth, min_samples_leaf):
    tree = DecisionTreeClassifier(random_state=seed, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X_train, y_train)
    leaf_sizes = tree.tree_.n_node_samples[tree.tree_.children_left == -1]
    return tree.get_depth(), tree.get_n_leaves(), leaf_sizes.mean(), (leaf_sizes == 1).mean()


if __name__ == "__main__":

    with open("configs/q2_finalized_params.json") as f:
        finalized_params = json.load(f)

    with open("configs/q1_finalized_params.json") as f:
        q1_finalized_params = json.load(f)

    single_tree_depth = q1_finalized_params["single_tree"]["max_depth"]
    single_tree_leaf = q1_finalized_params["single_tree"]["min_samples_leaf"]

    tree_depth = finalized_params["bagging"]["max_depth"]
    tree_leaf = finalized_params["bagging"]["min_sample_leaf"]
    bagging_n_trees = finalized_params["bagging"]["n_trees"]
    boosting_rounds = finalized_params["boosting"]["n_stumps"]
    boosting_depth = finalized_params["boosting"]["depth"]

    print(f"Selected single tree: max_depth={single_tree_depth}, min_samples_leaf={single_tree_leaf}")
    print(f"Final bagged model: {bagging_n_trees} trees with the same tree setting")
    print(f"Selected boosting: {boosting_rounds} rounds, depth {boosting_depth}")

    rng = np.random.default_rng([seed, 3]) 
    resampled_sets = make_resampled_indexes(y_train, num_repeats, rng)
    np.save("configs/q3_resampled_indexes.npy", resampled_sets)
    print(f"Resampled training sets: {resampled_sets.shape} (saved to configs/q3_resampled_indexes.npy)\n")

    make_tree = lambda r: DecisionTreeClassifier(random_state=seed, max_depth=single_tree_depth,
                                                 min_samples_leaf=single_tree_leaf)
    make_bagging = lambda r: bagged_trees(n_trees=bagging_n_trees, max_depth=tree_depth, min_sample_leaf=tree_leaf,
                                          rng=np.random.default_rng([seed, bagging_n_trees, r]))
    make_boosting = lambda r: SAMMEAdaboost(n_stumps=boosting_rounds, depth=boosting_depth)

    print("====================================================================================")
    print("#                              Question 3 ( Part a )")
    print("====================================================================================\n")

    t_0 = time.perf_counter()
    tree_probs, tree_accs, _ = run_refits(make_tree, tree_proba, resampled_sets)
    print(f"Single tree refits done ({time.perf_counter() - t_0:.1f}s)")
    t_0 = time.perf_counter()
    bagging_probs, bagging_accs, _ = run_refits(make_bagging, bagging_proba, resampled_sets)
    print(f"Bagging refits done ({time.perf_counter() - t_0:.1f}s)")
    t_0 = time.perf_counter()
    boosting_probs, boosting_accs, _ = run_refits(make_boosting, boosting_proba, resampled_sets)
    print(f"Boosting refits done ({time.perf_counter() - t_0:.1f}s)\n")

    tree_bias, tree_var = bias_variance(tree_probs, y_test)
    bagging_bias, bagging_var = bias_variance(bagging_probs, y_test)
    boosting_bias, boosting_var = bias_variance(boosting_probs, y_test)

    print(f"Single Tree -> Bias^2: {tree_bias:.4f} || Variance: {tree_var:.4f} || "
          f"Mean Acc: {tree_accs.mean():.4f} || Std Acc: {tree_accs.std():.4f}")
    print(f"Bagging -> Bias^2: {bagging_bias:.4f} || Variance: {bagging_var:.4f} || "
          f"Mean Acc: {bagging_accs.mean():.4f} || Std Acc: {bagging_accs.std():.4f}")
    print(f"Boosting -> Bias^2: {boosting_bias:.4f} || Variance: {boosting_var:.4f} || "
          f"Mean Acc: {boosting_accs.mean():.4f} || Std Acc: {boosting_accs.std():.4f}")

    print("\n====================================================================================")
    print("#                              Question 3 ( Part b )")
    print("====================================================================================\n")

    # the selected tree is already fully grown (max_depth=None), so the much deeper
    # tree is grown until every leaf is pure (min_samples_leaf=1)
    partb_settings = [
        ("Shallow", 8, tree_leaf),
        ("Selected", tree_depth, tree_leaf),
        ("Much deeper", None, 1),
    ]
    partb_n_trees = 100

    depth_results = []
    for name, max_depth, min_leaf in partb_settings:
        t_0 = time.perf_counter()
        tree_p, tree_a, _ = run_refits(
            lambda r, d=max_depth, l=min_leaf: DecisionTreeClassifier(random_state=seed, max_depth=d, min_samples_leaf=l),
            tree_proba, resampled_sets)
        bag_p, bag_a, bag_disagree = run_refits(
            lambda r, d=max_depth, l=min_leaf: bagged_trees(n_trees=partb_n_trees, max_depth=d, min_sample_leaf=l,
                                                            rng=np.random.default_rng([seed, partb_n_trees, r])),
            bagging_proba, resampled_sets)

        t_bias, t_var = bias_variance(tree_p, y_test)
        b_bias, b_var = bias_variance(bag_p, y_test)
        depth_reached, n_leaves, mean_leaf, one_sample_leaf = leaf_occupancy(max_depth, min_leaf)

        depth_results.append({
            "name": name, "max_depth": max_depth, "min_leaf": min_leaf, "depth_reached": depth_reached,
            "n_leaves": n_leaves, "mean_leaf": mean_leaf, "one_sample_leaf": one_sample_leaf,
            "tree_bias": t_bias, "tree_var": t_var, "bag_bias": b_bias, "bag_var": b_var,
            "var_removed": t_var - b_var, "var_removed_pct": 100 * (t_var - b_var) / t_var,
            "tree_acc": tree_a.mean(), "bag_acc": bag_a.mean(),
            "improvement": bag_a.mean() - tree_a.mean(), "disagreement": bag_disagree.mean(),
        })
        print(f"{name} (max_depth={max_depth}, min_samples_leaf={min_leaf}) done ({time.perf_counter() - t_0:.1f}s)")

    for res in depth_results:
        print(f"\nSetting: {res['name']} (max_depth={res['max_depth']}, min_samples_leaf={res['min_leaf']})")
        print(f"Single Tree Bias^2: {res['tree_bias']:.4f} || Single Tree Variance: {res['tree_var']:.4f}")
        print(f"Bagging Bias^2: {res['bag_bias']:.4f} || Bagging Variance: {res['bag_var']:.4f}")
        print(f"Variance removed by bagging: {res['var_removed']:.4f} ({res['var_removed_pct']:.1f}%)")
        print(f"Depth reached: {res['depth_reached']} || Number of Leaves: {res['n_leaves']} || "
              f"Average Samples per Leaf: {res['mean_leaf']:.1f} || "
              f"Fraction of Leaves with One Sample: {res['one_sample_leaf']:.3f}")
        print(f"Single Tree Acc: {res['tree_acc']:.4f} || Bagging Acc: {res['bag_acc']:.4f} || "
              f"Improvement: {res['improvement']:.4f} || Disagreement: {res['disagreement']:.4f}")

    print("\n====================================================================================")
    print("#                              Question 3 ( Part c )")
    print("====================================================================================\n")

    deepest = max(depth_results, key=lambda res: res["depth_reached"])
    best_improvement = max(depth_results, key=lambda res: res["improvement"])

    for label, res in (("Deepest tree", deepest), ("Largest bagging improvement", best_improvement)):
        print(f"{label}: {res['name']} (max_depth={res['max_depth']}, min_samples_leaf={res['min_leaf']}, "
              f"depth reached={res['depth_reached']})")
        print(f"Tree Bias^2: {res['tree_bias']:.4f} || Tree Variance: {res['tree_var']:.4f} || "
              f"Variance removed by bagging: {res['var_removed']:.4f} ({res['var_removed_pct']:.1f}%)")
        print(f"Average Samples per Leaf: {res['mean_leaf']:.1f} || Fraction of Leaves with One Sample: "
              f"{res['one_sample_leaf']:.3f} || Disagreement: {res['disagreement']:.4f}")
        print(f"Tree Acc: {res['tree_acc']:.4f} || Bagging Acc: {res['bag_acc']:.4f} || "
              f"Improvement: {res['improvement']:.4f}\n")

    print(f"Deepest tree is the depth with the largest bagging improvement: {deepest['name'] == best_improvement['name']}")

    print("\n====================================================================================")
    print("#                              Question 3 ( Part d )")
    print("====================================================================================\n")

    tree_final = make_tree(0).fit(X_train, y_train)
    bagging_final = bagged_trees(n_trees=bagging_n_trees, max_depth=tree_depth, min_sample_leaf=tree_leaf,
                                 rng=np.random.default_rng([seed, bagging_n_trees])).fit(X_train, y_train)
    boosting_final = make_boosting(0).fit(X_train, y_train)

    rows = [
        ("Single Tree", tree_final, tree_bias, tree_var, tree_accs),
        ("Bagging", bagging_final, bagging_bias, bagging_var, bagging_accs),
        ("Boosting", boosting_final, boosting_bias, boosting_var, boosting_accs),
    ]

    summary = []
    for name, model, bias, var, accs in rows:
        clean_acc = accuracy_score(y_test, model.predict(X_test))
        masked_acc = accuracy_score(y_test, model.predict(test_masked))
        summary.append({"name": name, "clean": clean_acc, "drop": clean_acc - masked_acc, "var": var})
        print(f"{name} -> Clean Acc: {clean_acc:.4f} || Masked Acc: {masked_acc:.4f} || "
              f"Drop: {clean_acc - masked_acc:.4f} || Bias^2: {bias:.4f} || Variance: {var:.4f} || "
              f"Std Acc (30 refits): {accs.std():.4f}")

    print(f"\nLowest variance: {min(summary, key=lambda s: s['var'])['name']}")
    print(f"Best clean accuracy: {max(summary, key=lambda s: s['clean'])['name']}")
    print(f"Largest masked drop: {max(summary, key=lambda s: s['drop'])['name']}")