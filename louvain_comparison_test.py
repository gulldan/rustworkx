import networkx as nx
import pandas as pd
import rustworkx as rx


def compare_louvain_implementations():
    """Детальное сравнение реализаций Louvain между NetworkX и rustworkx"""
    
    # Создаем граф клуба каратэ
    nx_graph = nx.karate_club_graph()
    rustworkx_graph = rx.networkx_converter(nx_graph)
    
    print("=== Сравнение алгоритмов Louvain ===")
    print(f"Граф: {nx_graph.number_of_nodes()} узлов, {nx_graph.number_of_edges()} рёбер")
    print()
    
    # Тестируем с разными параметрами
    resolutions = [0.5, 1.0, 1.5, 2.0]
    
    for resolution in resolutions:
        print(f"--- Resolution = {resolution} ---")
        
        # NetworkX Louvain
        try:
            nx_communities = nx.community.louvain_communities(nx_graph, resolution=resolution, seed=42)
            nx_modularity = nx.community.modularity(nx_graph, nx_communities, resolution=resolution)
            print(f"NetworkX: {len(nx_communities)} сообществ, модульность = {nx_modularity:.6f}")
            print(f"  Размеры сообществ: {[len(c) for c in nx_communities]}")
        except Exception as e:
            print(f"NetworkX ошибка: {e}")
        
        # RustworkX Louvain
        try:
            rx_communities = rx.community.louvain_communities(rustworkx_graph, resolution=resolution, seed=42)
            rx_modularity = rx.community.modularity(rustworkx_graph, rx_communities, resolution=resolution)
            print(f"RustworkX: {len(rx_communities)} сообществ, модульность = {rx_modularity:.6f}")
            print(f"  Размеры сообществ: {[len(c) for c in rx_communities]}")
        except Exception as e:
            print(f"RustworkX ошибка: {e}")
        
        print()

def test_simple_graph():
    """Тест на простом графе"""
    print("=== Тест на простом графе ===")
    
    # Создаем простой граф с явными сообществами
    G = nx.Graph()
    
    # Первое сообщество (треугольник)
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    # Второе сообщество (треугольник)
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])
    
    # Слабая связь между сообществами
    G.add_edge(2, 3)
    
    print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    
    # Конвертируем в rustworkx
    rx_graph = rx.networkx_converter(G)
    
    # NetworkX
    nx_communities = nx.community.louvain_communities(G, seed=42)
    nx_modularity = nx.community.modularity(G, nx_communities)
    print(f"NetworkX: {len(nx_communities)} сообществ, модульность = {nx_modularity:.6f}")
    print(f"  Сообщества: {[sorted(list(c)) for c in nx_communities]}")
    
    # RustworkX
    rx_communities = rx.community.louvain_communities(rx_graph, seed=42)
    rx_modularity = rx.community.modularity(rx_graph, rx_communities)
    print(f"RustworkX: {len(rx_communities)} сообществ, модульность = {rx_modularity:.6f}")
    print(f"  Сообщества: {[sorted(c) for c in rx_communities]}")
    
    print()

def debug_modularity_calculation():
    """Отладка расчета модульности"""
    print("=== Отладка расчета модульности ===")
    
    # Простой граф
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    rx_graph = rx.networkx_converter(G)
    
    # Попробуем разные разделения
    partitions = [
        [[0, 1, 2], [3, 4, 5]],  # Идеальное разделение
        [[0, 1, 2, 3], [4, 5]],  # Неидеальное разделение 1
        [[0, 1], [2, 3, 4, 5]],  # Неидеальное разделение 2
    ]
    
    for i, partition in enumerate(partitions):
        print(f"Разделение {i+1}: {partition}")
        
        # NetworkX
        nx_mod = nx.community.modularity(G, partition)
        print(f"  NetworkX модульность: {nx_mod:.6f}")
        
        # RustworkX
        rx_mod = rx.community.modularity(rx_graph, partition)
        print(f"  RustworkX модульность: {rx_mod:.6f}")
        
        # Разность
        diff = abs(nx_mod - rx_mod)
        print(f"  Разность: {diff:.6f}")
        print()

def compare_on_large_graph(file_path):
    """Сравнение на большом графе из Parquet файла"""
    print(f"=== Сравнение на большом графе: {file_path} ===")
    
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Ошибка: файл не найден по пути '{file_path}'.")
        print("Пожалуйста, убедитесь, что файл существует и путь указан верно.")
        print()
        return

    # Создаем граф NetworkX
    G_nx = nx.from_pandas_edgelist(df, "src", "dst")
    print(f"Граф загружен: {G_nx.number_of_nodes()} узлов, {G_nx.number_of_edges()} рёбер")
    
    # Конвертируем в rustworkx
    G_rx = rx.networkx_converter(G_nx, keep_attributes=False) # Атрибуты не нужны для Louvain
    
    resolution = 1.0 # Используем стандартное разрешение
    print(f"--- Resolution = {resolution} ---")
    
    # NetworkX Louvain
    try:
        print("Запуск NetworkX Louvain...")
        nx_communities = nx.community.louvain_communities(G_nx, resolution=resolution, seed=42)
        nx_modularity = nx.community.modularity(G_nx, nx_communities, resolution=resolution)
        print(f"NetworkX: {len(nx_communities)} сообществ, модульность = {nx_modularity:.6f}")
        # Вывод размеров сообществ для больших графов может быть громоздким, поэтому закомментируем
        # print(f"  Размеры сообществ: {[len(c) for c in nx_communities]}")
    except Exception as e:
        print(f"NetworkX ошибка: {e}")
        
    # RustworkX Louvain
    try:
        print("Запуск RustworkX Louvain...")
        rx_communities = rx.community.louvain_communities(G_rx, resolution=resolution, seed=42)
        rx_modularity = rx.community.modularity(G_rx, rx_communities, resolution=resolution)
        print(f"RustworkX: {len(rx_communities)} сообществ, модульность = {rx_modularity:.6f}")
        # print(f"  Размеры сообществ: {[len(c) for c in rx_communities]}")
    except Exception as e:
        print(f"RustworkX ошибка: {e}")
        
    print()


if __name__ == "__main__":
    test_simple_graph()
    debug_modularity_calculation()
    compare_louvain_implementations()
    # Укажите путь к вашему файлу
    large_graph_path = "datasets/wiki_news_edges_sim_thresh_0_9.parquet"
    compare_on_large_graph(large_graph_path) 