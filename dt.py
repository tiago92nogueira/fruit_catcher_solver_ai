import csv
import math
from collections import Counter, defaultdict

class DecisionTree:
    def __init__(self, dataset, feature_names, target_label):
        
        self.tree = self._build_tree(dataset, feature_names, target_label)

    def _entropy(self, dataset, target_label):
        # calcula a entropia com base na distribuição das classes
        label_counts = Counter([item[target_label] for item in dataset])
        total = len(dataset)
        entropy = 0
        for count in label_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    def _information_gain(self, dataset, feature, target_label):
        # calcula o ganho de informação relacionado com um atributo
        base_entropy = self._entropy(dataset, target_label)
        total = len(dataset)

        # agrupa os dados por valor do atributo
        partitions = defaultdict(list)
        for item in dataset:
            partitions[item[feature]].append(item)

        weighted_entropy = sum(
            (len(subset) / total) * self._entropy(subset, target_label)
            for subset in partitions.values()
        ) # entropia ponderada dos subconjuntos
        return base_entropy - weighted_entropy

    def _most_common_label(self, dataset, target_label):
        #valor de classe mais frequente
        return Counter(item[target_label] for item in dataset).most_common(1)[0][0]

    def _build_tree(self, dataset, feature_names, target_label):
        # verifica se todos os exemplos tem a mesma classe
        class_values = [item[target_label] for item in dataset]
        if len(set(class_values)) == 1:
            return class_values[0]  #todos os rótulos são iguais

        # nenhum atributo restante para dividir
        if not feature_names:
            return self._most_common_label(dataset, target_label)

        # escolhe o melhor atributo com base no ganho de informação
        gains = {feature: self._information_gain(dataset, feature, target_label)
                 for feature in feature_names}
        best_feature = max(gains, key=gains.get)

        # cria um nó para o melhor atributo
        tree_node = {best_feature: {}}
        feature_values = set(item[best_feature] for item in dataset)

        for value in feature_values:
            # filtra os dados com base no valor atual do atributo
            subset = [item for item in dataset if item[best_feature] == value]
            if not subset:
                # caso nenhum exemplo seja bom usa a classe mais comum
                tree_node[best_feature][value] = self._most_common_label(dataset, target_label)
            else:
                # continua a recursão e remove o atributo usado
                remaining_features = [f for f in feature_names if f != best_feature]
                subtree = self._build_tree(subset, remaining_features, target_label)
                tree_node[best_feature][value] = subtree

        return tree_node

    def predict(self, item, tree=None):
        # preve com base na dt
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return tree  

        current_attr = next(iter(tree))
        item_value = item.get(current_attr)

        if item_value in tree[current_attr]:
            return self.predict(item, tree[current_attr][item_value])
        else:
            return -1  

def load_csv(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            try:
                # converte is_fruit para int
                row['is_fruit'] = int(row['is_fruit'].strip())

                for key in row:
                    if isinstance(row[key], str):
                        row[key] = row[key].strip()

                data.append(row)
            except (ValueError, KeyError):
                print(f"Ignorando linha inválida: {row}")
    return data

def train_decision_tree(csv_file_path):
    dataset = load_csv(csv_file_path)
    features = ['name', 'color', 'format']  # atributos 
    model = DecisionTree(dataset, features, target_label='is_fruit')
    return model
