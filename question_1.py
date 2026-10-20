import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from data_setup import X_train, y_train, X_val, y_val, X_test, y_test,test_masked, seed
import json


def bootstrap_sample(X, y, rng):
    n_samples = X.shape[0]
    indexes = rng.choice(n_samples, size = n_samples, replace = True)
    return indexes, X[indexes], y[indexes]

class bagged_trees:
    def __init__(self, n_trees, max_depth, min_sample_leaf, rng):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.trees = []
        self.bags = []
        self.avg_distinct_samples = 0
        self.rng = rng

    def fit(self, X, y):
        total_distinct = 0
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(random_state=seed, max_depth= self.max_depth, min_samples_leaf= self.min_sample_leaf)
            indexes, X_bootstrap, y_bootstrap = bootstrap_sample(X, y, self.rng)
            n_distinct_examples = len(np.unique(indexes))

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            self.bags.append(indexes)

            total_distinct += n_distinct_examples

        self.avg_distinct_samples = total_distinct / self.n_trees
        return self


    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_predictions, 0, 1)
        y_preds = np.array([np.bincount(preds).argmax() for preds in tree_preds])
        return y_preds

def tree_conf(model, X):      
    return model.predict_proba(X).max(axis=1).mean()

def bagging_conf(model, X):
    preds = np.array([t.predict(X) for t in model.trees])   
    final = model.predict(X)
    return (preds == final).mean(axis=0).mean()

def oob_accuracy(model, X, y):
    n = X.shape[0]
    num_classes = 40
    votes = np.zeros((n, num_classes))
    for tree, bag in zip(model.trees, model.bags):
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(bag)] = False         
        if oob_mask.sum() == 0:
            continue
        preds = tree.predict(X[oob_mask])
        for i, p in zip(np.where(oob_mask)[0], preds):
            votes[i, p] += 1
    voted = votes.sum(axis=1) > 0            
    oob_pred = np.argmax(votes, axis=1)
    return accuracy_score(y[voted], oob_pred[voted])


def disagreement_and_margin(model, X):
    preds = np.array([t.predict(X) for t in model.trees])   # (T, n)
    T, n = preds.shape
    num_classes = 40

    total, pairs = 0.0, 0
    for i in range(T):
        for j in range(i + 1, T):
            total += np.mean(preds[i] != preds[j])
            pairs += 1
    disagreement = total / pairs if pairs > 0 else 0.0

    margins = []
    for s in range(n):
        counts = np.bincount(preds[:, s], minlength=num_classes) / T
        top2 = np.sort(counts)[::-1]
        margins.append(top2[0] - top2[1])
    return disagreement, float(np.mean(margins))

def bagging_votes(model, X):
    preds = np.array([t.predict(X) for t in model.trees])     
    num_classes = 40
    return np.array([np.bincount(preds[:, s], minlength=num_classes)
                     for s in range(X.shape[0])])

if __name__ == "__main__":

    print("====================================================================================")
    print("#                              Question 1 ( Part a )")
    print("====================================================================================\n")

    max_depths = [2,4,8,16, None]
    min_samples_leafs = [1, 2, 4]

    best_acc, best_params, best_tree = 0, None, None

    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leafs:

            baseline_tree = DecisionTreeClassifier(random_state = seed, 
                                                    max_depth=max_depth, 
                                                    min_samples_leaf=min_samples_leaf )
            baseline_tree.fit(X_train, y_train)
            y_pred = baseline_tree.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"Max Depth: {max_depth}, Min Samples Leaf: {min_samples_leaf}, Validation Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
                best_tree = baseline_tree


    print(f"\n\n==================== Best Baseline Decision Tree Params =====================")
    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    print(f"Best Parameters: Max Depth: {best_params[0]}, Min Samples Leaf: {best_params[1]}")

    train_acc = accuracy_score(y_train, best_tree.predict(X_train))
    val_acc = accuracy_score(y_val, best_tree.predict(X_val))
    test_acc = accuracy_score(y_test, best_tree.predict(X_test))
    num_leaves = best_tree.get_n_leaves()
    avg_samples_per_leaf = best_tree.tree_.n_node_samples[best_tree.tree_.children_left == -1].mean()
    leaves_with_one_sample = (best_tree.tree_.n_node_samples[best_tree.tree_.children_left == -1] == 1).mean()

    print(f"\n\n==================== Best Baseline Decision Tree Performance =====================")
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Number of Leaves: {num_leaves}")
    print(f"Average Samples per Leaf: {avg_samples_per_leaf:.1f}")
    print(f"Fraction of Leaves containing One Sample: {leaves_with_one_sample:.3f}")

    print(f"\n\n==================== Decision Tree Performance for Different Depths =====================")
    for d in [2, 4, 8, 16, None]:
        d_tree = DecisionTreeClassifier(random_state=seed, max_depth=d,
                                min_samples_leaf=best_params[1])
        d_tree.fit(X_train, y_train)                          
        ls = d_tree.tree_.n_node_samples[d_tree.tree_.children_left == -1]
        print(f"\nDepth: {d}")
        print(f"Training Accuracy: {round(accuracy_score(y_train, d_tree.predict(X_train)), 3)} ||",
            f"Validation Accuracy: {round(accuracy_score(y_val, d_tree.predict(X_val)), 3)} ||",
            f"Test Accuracy: {round(accuracy_score(y_test, d_tree.predict(X_test)), 3)}",
            f"\nNumber of Leaves: {d_tree.get_n_leaves()} ||",
            f"Average Samples per Leaf: {round(ls.mean(), 1)} ||",
            f"Fraction of Leaves with One Sample: {round((ls == 1).mean(), 3)}")


    print("====================================================================================")
    print("#                              Question 1 ( Part b )")
    print("====================================================================================\n")


    allowed_n_trees = [1, 5, 10, 25, 50, 100, 200]
    avg_distinct_samples_last = 0
    print(f"\n============================ Results for Bagged Trees with Different Number of Trees ============================")

    for n_trees in allowed_n_trees:
        bagged_tree = bagged_trees(n_trees= n_trees, max_depth= best_params[0], min_sample_leaf= best_params[1], rng = np.random.default_rng([seed, n_trees]))
        bagged_tree.fit(X_train, y_train)

        print(f"\nNumber of Trees: {n_trees}")
        train_preds = bagged_tree.predict(X_train)
        val_preds =  bagged_tree.predict(X_val)
        test_preds = bagged_tree.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_preds)
        val_accuracy = accuracy_score(y_val, val_preds)
        test_accuracy = accuracy_score(y_test, test_preds)

        if n_trees == allowed_n_trees[-1]:
            avg_distinct_samples_last = bagged_tree.avg_distinct_samples

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")       
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Number of distinct examples in bootstrap sample: {bagged_tree.avg_distinct_samples:.4f}")

    n = len(X_train)
    expected = n * (1 - (1 - 1/n) ** n)  

    print(f"\n==================== Comparison Between theoretical and observed (B = 200) ============================")

    print(f"\nTheoretical number of distinct examples in bootstrap sample: {expected:.4f}")
    print(f"Average number of distinct examples in bootstrap sample for 200 trees: {avg_distinct_samples_last:.4f}")
    print(f"Difference between theoretical and observed: {abs(expected - avg_distinct_samples_last):.4f}")



    print("====================================================================================")
    print("#                              Question 1 ( Part c )")
    print("====================================================================================\n")
    
    allowed_n_trees = [1, 5, 10, 25, 50, 100, 200]

    oob_list, disagree_list, margin_list, val_list = [], [], [], []

    max_val = 0
    ensemble_size = None
    part_c_best_model = None

    for n_trees in allowed_n_trees:
        model = bagged_trees(n_trees=n_trees, max_depth=best_params[0], min_sample_leaf=best_params[1], rng=np.random.default_rng([seed, n_trees]))
        model.fit(X_train, y_train)

        oob = oob_accuracy(model, X_train, y_train)
        disagreement, margin = disagreement_and_margin(model, X_val)   # eval on validation
        val_acc = accuracy_score(y_val, model.predict(X_val))

        oob_list.append(oob)
        disagree_list.append(disagreement)
        margin_list.append(margin)
        val_list.append(val_acc)

        if val_acc > max_val:
            max_val = val_acc
            ensemble_size = n_trees
            part_c_best_model = model

        print(f"B = {n_trees  } | OOB = {oob:.4f} | disagreement = {disagreement:.4f} | margin = {margin:.4f} | val = {val_acc:.4f}")

    print("\n======================Best Ensemble Size and Validation Accuracy=====================================")
    print(f"\nBest Validation Accuracy (part c): {max_val}")
    print(f"\nBest Ensemble Size (part c): {ensemble_size}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(allowed_n_trees, oob_list, marker="o"); ax[0].set_title("OOB accuracy")
    ax[1].plot(allowed_n_trees, disagree_list, marker="o"); ax[1].set_title("Avg pairwise disagreement")
    ax[2].plot(allowed_n_trees, margin_list, marker="o"); ax[2].set_title("Mean vote margin")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(allowed_n_trees); a.set_xticklabels(allowed_n_trees)
        a.set_xlabel("B (number of trees)")
    ax[0].set_ylabel("OOB Accuracy")
    ax[1].set_ylabel("disagreement")
    ax[2].set_ylabel("Vote margin")
    plt.savefig("figures/question1_partc.png", dpi=150, bbox_inches="tight")
    plt.close()

    finalized_params = {
        "single_tree": {
            "max_depth": best_params[0],
            "min_samples_leaf": best_params[1],
        },
        "bagging": {
            "n_trees": ensemble_size,
            "max_depth": best_params[0],
            "min_sample_leaf": best_params[1],
        },
    }
    with open("configs/q1_finalized_params.json", "w") as f:
        json.dump(finalized_params, f, indent=2)

    print("====================================================================================")
    print("#                              Question 1 ( Part d )")
    print("====================================================================================\n")
    
    single_tree_clean_preds = best_tree.predict(X_test)
    single_tree_clean_acc = accuracy_score(y_test, single_tree_clean_preds)
    bagging_clean_preds = part_c_best_model.predict(X_test)
    bagging_clean_acc = accuracy_score(y_test, bagging_clean_preds)


    print(f"\n====================Predictions on Clean Testset===========================\n")
    print(f"Single Tree Accuracy: {single_tree_clean_acc}")
    print(f"Single Tree Mean Winning Prob: {tree_conf(best_tree, X_test)}\n")
    print(f"Bagging Model Accuracy: {bagging_clean_acc}")
    print(f"Bagging Model Mean Winning Prob: {bagging_conf(part_c_best_model, X_test)}\n")

    single_tree_masked_preds = best_tree.predict(test_masked)
    bagging_masked_preds = part_c_best_model.predict(test_masked)

    single_tree_masked_acc = accuracy_score(y_test, single_tree_masked_preds)
    bagging_masked_acc = accuracy_score(y_test, bagging_masked_preds)
   
    print(f"\n=======================Predictions on Masked Testset=================================\n")
    print(f"Single Tree Accuracy: {single_tree_masked_acc}")
    print(f"Single Tree Mean Winning Prob: {tree_conf(best_tree, test_masked)}\n")
    print(f"Bagging Model Accuracy: {bagging_masked_acc}")
    print(f"Bagging Model Mean Winning Prob: {bagging_conf(part_c_best_model, test_masked)}\n")


    tree_pred = best_tree.predict(X_test)
    votes = bagging_votes(part_c_best_model, X_test)
    bag_pred  = votes.argmax(axis=1)                              

    corrects = np.where((tree_pred != y_test) & (bag_pred == y_test))[0]  
    both_wrong = np.where((tree_pred != y_test) & (bag_pred != y_test))[0]  

    def show(i):
        nz = np.nonzero(votes[i])[0]                           
        vote_str = {int(c): int(votes[i, c]) for c in nz}
        print(f"  idx {i}: true={y_test[i]}, tree={tree_pred[i]}, bagging={bag_pred[i]}, votes={vote_str}")

    print("\nWhere Bagging corrects the tree:")
    for i in corrects[:2]:
        show(i)

    print("\nWhere both are wrong:")
    for i in both_wrong[:2]:
        show(i)


    faces = list(corrects[:2]) + list(both_wrong[:2])
    titles = ["Bagging fixes", "Bagging fixes", "Both wrong", "Both wrong"]

    plt.figure(figsize=(12, 4))
    for k, i in enumerate(faces):
        plt.subplot(1, 4, k + 1)
        plt.imshow(X_test[i].reshape(64, 64), cmap="gray")
        plt.title(f"{titles[k]}\ntrue={y_test[i]} tree={tree_pred[i]} bag={bag_pred[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("figures/question1_partd_faces.png", dpi=150)
    plt.close()

