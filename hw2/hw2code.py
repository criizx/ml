from sklearn.base import BaseEstimator, ClassifierMixin  # Добавьте этот импорт, если его нет

class DecisionTree(BaseEstimator, ClassifierMixin):  # Измените на это (добавьте наследование)
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    # Добавьте эти два метода:
    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # Остальной код класса остается без изменений (def _fit_node, def _predict_node, def fit, def predict)
    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        feature_best = None
        threshold_best = None
        gini_best = np.inf
        split_best = None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key in counts:
                    ratio[key] = clicks[key] / counts[key] if counts[key] > 0 else 0
                sorted_categories = [c for c, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue
            if gini < gini_best:
                gini_best = gini
                feature_best = feature
                if feature_type == "real":
                    threshold_best = threshold
                    split_best = (feature_vector <= threshold)
                else:
                    threshold_best = [c for c in sorted_categories if categories_map[c] < threshold]
                    split_best = np.array([x in threshold_best for x in sub_X[:, feature]])
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best
        node["left_child"] = {}
        node["right_child"] = {}
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"])
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature_idx = node["feature_split"]
        if self._feature_types[feature_idx] == "real":
            if x[feature_idx] <= node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)