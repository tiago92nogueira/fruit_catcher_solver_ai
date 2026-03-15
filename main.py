import random
import argparse
from pathlib import Path

from game import start_game, get_score
from genetic import genetic_algorithm
from nn import create_network_architecture
from dt import train_decision_tree

STATE_SIZE = 1 + 3 * 3  # basket_y + (x, y, is_fruit) * 3
MAX_SCORE = 100         # valor arbitrário alvo para fitness

def fitness(nn, individual, seed):
    nn.load_weights(individual)
    random.seed(seed)
    return get_score(player=lambda state: nn.forward(state))

def train_ai_player(filename, population_size, generations):
    print("Iniciando treino da IA...")
    nn = create_network_architecture(STATE_SIZE)
    individual_size = nn.compute_num_weights()

    fitness_function = lambda individual, seed=None: fitness(nn, individual, seed)

    # função para mostrar o melhor fitness de cada geração
    def on_generation(gen, best_fit):
        print(f"Geração {gen}: Melhor fitness = {best_fit:.2f}")

    best, best_fitness = genetic_algorithm(
        individual_size,
        population_size,
        fitness_function,
        MAX_SCORE,
        generations,
        on_generation=on_generation  # Passa o callback
    )

    with open(filename, 'w') as f:
        f.write(','.join(map(str, best)))

    print(f"Treino concluído. Melhor indivíduo guardado em '{filename}'.")
    print(f"Melhor fitness final: {best_fitness:.2f}")


def load_ai_player(filename):
    file_path = Path(filename)
    if not file_path.exists():
        print(f"Arquivo '{filename}' não encontrado.")
        return None

    with open(filename, 'r') as f:
        weights = list(map(float, f.read().split(',')))

    nn = create_network_architecture(STATE_SIZE)
    nn.load_weights(weights)

    return lambda state: nn.forward(state)

def train_fruit_classifier(filename):
    dt = train_decision_tree(filename)
    attributes = ['name', 'color', 'format']

    def classifier(item_list):
        # Transformar a lista em dict para o DT
        item_dict = dict(zip(attributes, item_list))
        return dt.predict(item_dict)

    return classifier


def main():
    parser = argparse.ArgumentParser(description='IA 2024/2025 - Projeto Fruit Catcher')
    parser.add_argument('-t', '--train', action='store_true', help='Treina a IA com algoritmo genético')
    parser.add_argument('-p', '--population', default=100, type=int, help='Tamanho da população')
    parser.add_argument('-g', '--generations', default=100, type=int, help='Número de gerações')
    parser.add_argument('-f', '--file', default='best_individual.txt', help='Arquivo para salvar/carregar pesos da IA')
    parser.add_argument('-l', '--headless', action='store_true', help='Executa sem interface gráfica')
    args = parser.parse_args()

    if args.train:
        train_ai_player(args.file, args.population, args.generations)
        return

    ai_player = load_ai_player(args.file)
    if ai_player is None:
        print("Falha ao carregar o jogador IA.")
        return

    fruit_classifier = train_fruit_classifier('train.csv')

    if args.headless:
        score = get_score(ai_player, fruit_classifier)
        print(f'Score final: {score}')
    else:
        start_game(ai_player, fruit_classifier)

if __name__ == '__main__':
    main()
