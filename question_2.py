import numpy as np
import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from data_setup import X_train, X_val, X_test, y_train, y_val, y_test, test_masked, seed
from question_1 import bagged_trees, tree_conf, bagging_conf, bagging_votes


finalized_params = {"bagging": {}, "boosting": {}}

class SAMMEAdaboost:

    def __init__(self, n_stumps: int, depth: int):
        self.n_stumps = n_stumps
        self.depth = depth
        self.classifiers = []
        self.weights_before = []
        self.weights_after = []
        self.errors = []
        self.round_preds = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.classes = np.unique(y)
        K = len(self.classes)

        weights = np.full(n_samples, 1 / n_samples)

        random_bound = 1 - 1 / K
        for _ in range(self.n_stumps):
            self.weights_before.append(weights.copy())

            clf = DecisionTreeClassifier(max_depth=self.depth, random_state=seed)
            clf.fit(X, y, sample_weight=weights)
            predictions = clf.predict(X)
            error = weights[predictions != y].sum()

            if error >= random_bound:
                print(f"Stopping early: learner error {error:.4f} >= random bound {random_bound:.4f} (alpha <= 0)")
                self.weights_before.pop()
                break

            error = max(error, 1e-10)
            alpha = np.log((1 - error) / error) + np.log(K - 1)
            incorrect = (predictions != y).astype(float)
            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()

            self.classifiers.append(clf)
            self.errors.append(error)
            self.round_preds.append(predictions.copy())
            self.alphas.append(alpha)
            self.weights_after.append(weights.copy())
        return self

    def decision_scores(self, X):
        num_classes = len(self.classes)
        S = np.zeros((X.shape[0], num_classes))
        for clf, alpha in zip(self.classifiers, self.alphas):
            preds = clf.predict(X)
            for i, c in enumerate(self.classes):
                S[preds == c, i] += alpha
        return S

    def predict(self, X):
        preds = self.decision_scores(X)
        return self.classes[np.argmax(preds, axis=1)]

    def confidence(self, X):
        scores = self.decision_scores(X)
        total_alpha = sum(self.alphas)
        return scores / total_alpha


def boosting_conf(model, X):
        return model.confidence(X).max(axis=1).mean()

if __name__=="__main__":

    print("====================================================================================")
    print(".                             Question 2 ( Part a )")
    print("====================================================================================\n")

    dummy_data = np.array([[1.5, 1.0], [2.0, 7.0], [1.0, 0.5], [2.0, 4.0], [1.5, 6.0],[3.0, 4.5], [1.0, 5.0], [1.0, 5.5], [1.5, 0.5], [3.5, 5.5]])
    dummy_label = np.array([0,2,0,1,2,1,2,2,0,1])

    model_1_round = SAMMEAdaboost(n_stumps=1, depth=1)
    model_1_round.fit(dummy_data, dummy_label)

    print(f"No. of Classes: {model_1_round.classes}")
    print(f"Number of samples: {len(dummy_label)}")

    print(f"Initial sample weights: {np.round(model_1_round.weights_before[0], 4)}")
    print(f"True labels:{dummy_label}")
    print(f"Weak learner predictions: {model_1_round.round_preds[0]}")
    print(f"Weighted error: {model_1_round.errors[0]:.4f}")
    print(f"Alpha: {model_1_round.alphas[0]:.4f}")
    print(f"Updated normalised weights: {np.round(model_1_round.weights_after[0], 4)}")
    print(f"Sum of updated weights: {model_1_round.weights_after[0].sum():.4f}")

    print("\n====================================================================================")
    print("                             Question 2 ( Part b )")
    print("====================================================================================\n")

    week_learner_depths = [1,2,3,4]
    rounds = [10, 25, 50, 100]
    best_val_acc, best_params  = 0, None
    best_model = None

    for n_round in rounds:
        for depth in week_learner_depths:
            model = SAMMEAdaboost(n_stumps = n_round, depth = depth)
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            print(f"\nFor No. of Rounds: {n_round} and Depth of weak learners: {depth} ")
            print(f"\nTrain Accuracy: {train_acc:.4f}")
            print(f"Val Accuracy: {val_acc}")
            print(f"Test Accuracy: {test_acc}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = n_round, depth
                best_model = model
    print("\n====================Val Accuracy and Params (part b)===================")
    print(f"\nBest Validation Accuracy: {best_val_acc}")
    print(f"No. of rounds: {best_params[0]}, Depth: {best_params[1]}")

    best_model_weighted_erros = np.round(best_model.errors, 4)
    print(best_model_weighted_erros)
    learner_weight = np.round(best_model.alphas, 4)
    max_weight = np.round([w.max() for w in best_model.weights_after],4)
    top_10_weight = np.round([np.sort(w)[::-1][:10].sum() for w in best_model.weights_after], 4)


    best_rounds = best_params[0]
    round_axis = range(1, best_rounds + 1)

    fig, ax = plt.subplots(2, 2, figsize = (12, 8))
    ax[0,0].plot(round_axis, best_model_weighted_erros)
    ax[0,0].set_title("Weighted Training Error")
    ax[0,1].plot(round_axis, learner_weight)
    ax[0,1].set_title("learner Weights")
    ax[1,0].plot(round_axis, max_weight)
    ax[1,0].set_title("Max Sample Weights")
    ax[1,1].plot(round_axis, top_10_weight)
    ax[1,1].set_title("Top 10 Weights")

    plt.savefig("figures/question2_partb_tracking.png", dpi=150)  
    plt.close()

    print("\n====================================================================================")
    print("                             Question 2 ( Part c )")
    print("====================================================================================\n")


    allowed_learners = [25, 50, 100]

    best_bagging_val, best_boosting_val = 0, 0
    best_bagging_model = None
    best_boosting_model = None
    
    print(f"\n============================ Results with Different Number of learners ============================")
    for num_learners in allowed_learners:

        bagged_tree = bagged_trees(n_trees= num_learners, max_depth= None, min_sample_leaf= 4, rng = np.random.default_rng([seed, num_learners]))
        t_0 = time.perf_counter()
        bagged_tree.fit(X_train, y_train)
        bag_time = time.perf_counter() - t_0

        boosting_tree = SAMMEAdaboost(n_stumps = num_learners, depth = best_params[1])
        t_0 = time.perf_counter()
        boosting_tree.fit(X_train, y_train)
        boost_time = time.perf_counter() - t_0

        print(f"\nNumber of Learners: {num_learners}\n")
        bagging_train_preds = bagged_tree.predict(X_train)
        bagging_val_preds =  bagged_tree.predict(X_val)
        bagging_test_preds = bagged_tree.predict(X_test)
        bagging_train_accuracy = accuracy_score(y_train, bagging_train_preds)
        bagging_val_accuracy = accuracy_score(y_val, bagging_val_preds)
        bagging_test_accuracy = accuracy_score(y_test, bagging_test_preds)
      
        print(f"Bagging Train Accuracy: {bagging_train_accuracy:.4f}")
        print(f"Bagging Validation Accuracy: {bagging_val_accuracy:.4f}")       
        print(f"Bagging Test Accuracy: {bagging_test_accuracy:.4f}")
        print(f"Bagging Training Time: {bag_time:.2f}s")

        boosting_train_preds = boosting_tree.predict(X_train)
        boosting_val_preds =  boosting_tree.predict(X_val)
        boosting_test_preds = boosting_tree.predict(X_test)
        boosting_train_accuracy = accuracy_score(y_train, boosting_train_preds)
        boosting_val_accuracy = accuracy_score(y_val, boosting_val_preds)
        boosting_test_accuracy = accuracy_score(y_test, boosting_test_preds)

        print(f"\nBoosting Train Accuracy: {boosting_train_accuracy:.4f}")
        print(f"Boosting Validation Accuracy: {boosting_val_accuracy:.4f}")       
        print(f"Boosting Test Accuracy: {boosting_test_accuracy:.4f}")
        print(f"Boosting Training Time: {boost_time:.2f}s")

    
        if bagging_val_accuracy > best_bagging_val:
            best_bagging_val = bagging_val_accuracy
            best_bagging_model = bagged_tree
 
            finalized_params["bagging"] = {
                "n_trees": best_bagging_model.n_trees,
                "max_depth": best_bagging_model.max_depth,
                "min_sample_leaf": best_bagging_model.min_sample_leaf,
            }
        

        if boosting_val_accuracy > best_boosting_val:
            best_boosting_val = boosting_val_accuracy
            best_boosting_model = boosting_tree

            finalized_params["boosting"] = {
                "n_stumps": best_boosting_model.n_stumps,
                "depth": best_boosting_model.depth,
            }

    final_weights_best_boosting_model = best_boosting_model.weights_after[-1]
    index = int(np.argmax(final_weights_best_boosting_model))

    pred_history = [int(preds[index]) for preds in best_boosting_model.round_preds]
    weight_history = [float(w[index]) for w in best_boosting_model.weights_after]

    print(f"\nHard face index: {index}")
    print(f"True label: {y_train[index]}")
    print(f"Prediction history: {pred_history}")
    print(f"Weight history: {np.round(weight_history, 4)}")
    print(f"Times misclassified: {sum(p != y_train[index] for p in pred_history)} / {len(pred_history)}")

    plt.figure()
    plt.imshow(X_train[index].reshape(64, 64), cmap="gray")
    plt.title(f"Hard face: true label {y_train[index]}")
    plt.axis("off")
    plt.savefig("figures/question2_hard_face.png", dpi=150, bbox_inches="tight")
    plt.close()
    
   
    print("\n====================================================================================")
    print("                             Question 2 ( Part d )")
    print("====================================================================================\n")

    with open("configs/q1_finalized_params.json") as f:
        q1_finalized_params = json.load(f)

    single_tree_depth = q1_finalized_params["single_tree"]["max_depth"]
    single_tree_leaf = q1_finalized_params["single_tree"]["min_samples_leaf"]
    
    single_tree = DecisionTreeClassifier(random_state=seed, max_depth=single_tree_depth,
                                min_samples_leaf= single_tree_leaf)
    single_tree.fit(X_train, y_train)


    single_tree_clean_preds = single_tree.predict(X_test)
    single_tree_clean_acc = accuracy_score(y_test, single_tree_clean_preds)
    bagging_clean_preds = best_bagging_model.predict(X_test)
    bagging_clean_acc = accuracy_score(y_test, bagging_clean_preds)
    boosting_clean_preds = best_boosting_model.predict(X_test)
    boosting_clean_acc = accuracy_score(y_test, boosting_clean_preds)


    print(f"\nPredictions on Clean Testset\n")
    print(f"Single Tree Accuracy: {single_tree_clean_acc}")
    print(f"Single Tree Mean Winning Prob: {tree_conf(single_tree, X_test)}\n")
    print(f"Bagging Model Accuracy: {bagging_clean_acc}")
    print(f"Bagging Model Mean Winning Prob: {bagging_conf(best_bagging_model, X_test)}\n")
    print(f"Boosting Model Accuracy: {boosting_clean_acc}")
    print(f"Boosting Model Mean Winning Prob: {boosting_conf(best_boosting_model, X_test)}\n")


    single_tree_masked_preds = single_tree.predict(test_masked)
    bagging_masked_preds = best_bagging_model.predict(test_masked)
    boosting_masked_preds = best_boosting_model.predict(test_masked)

    single_tree_masked_acc = accuracy_score(y_test, single_tree_masked_preds)
    bagging_masked_acc = accuracy_score(y_test, bagging_masked_preds)
    boosting_masked_acc = accuracy_score(y_test, boosting_masked_preds)

    print(f"\n Predictions on Masked Testset\n")
    print(f"Single Tree Accuracy: {single_tree_masked_acc}")
    print(f"Single Tree Mean Winning Prob: {tree_conf(single_tree, test_masked)}\n")
    print(f"Bagging Model Accuracy: {bagging_masked_acc}")
    print(f"Bagging Model Mean Winning Prob: {bagging_conf(best_bagging_model, test_masked)}\n")
    print(f"Boosting Model Accuracy: {boosting_masked_acc}")
    print(f"Boosting Model Mean Winning Prob: {boosting_conf(best_boosting_model, test_masked)}\n")

    possible_index = np.where((bagging_clean_preds == boosting_clean_preds) & (bagging_masked_preds   != boosting_masked_preds))[0]
    index = possible_index[0]

    print(f"Face index: {index}, true label: {y_test[index]}")
    print(f"Clean Images : Bagging: {bagging_clean_preds[index]} ||  Boosting: {boosting_clean_preds[index]}")
    print(f"Masked Images: Bagging: {bagging_masked_preds[index]} ||  Boosting: {boosting_masked_preds[index]}")

    plt.figure()
    plt.imshow(test_masked[index].reshape(64, 64), cmap = 'gray')
    plt.title(f"true: {y_test[index]} | bag: {bagging_masked_preds[index]} | boost: {boosting_masked_preds[index]}")
    plt.axis("off")
    plt.savefig("figures/question2_disagreeement_face.png", dpi = 150, bbox_inches = 'tight')
    plt.close()

    with open("configs/q2_finalized_params.json", "w") as f:
        json.dump(finalized_params, f, indent = 2)
