from dt import train_decision_tree

tree = train_decision_tree("train.csv")

exemplos = [
    {"name": "banana", "color": "blue", "format": "curved"},
    {"name": "apple", "color": "red", "format": "circle"},
    {"name": "pear", "color": "blue", "format": "oval"}
]

for ex in exemplos:
    resultado = tree.predict(ex)
    print(f"Exemplo: {ex} → Classificação: {resultado}")
