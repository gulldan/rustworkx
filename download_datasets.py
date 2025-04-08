#!/usr/bin/env python3

import os
import shutil
import tarfile
import tempfile
import urllib.request

import networkx as nx


def download_file(url, filename):
    """Download a file from URL if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

def download_from_github(repo_path, filename, output_path):
    """Download a file from GitHub repository."""
    if not os.path.exists(output_path):
        url = f"https://raw.githubusercontent.com/vlivashkin/community-graphs/master/{repo_path}/{filename}"
        print(f"Downloading {output_path} from GitHub...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path}")
    else:
        print(f"{output_path} already exists")

def save_football_network():
    """Download and save the football network in GML format."""
    if not os.path.exists("datasets/football.gml"):
        # Используем репозиторий vlivashkin/community-graphs
        download_from_github("gml_graphs", "football.gml", "datasets/football.gml")
        
        # Проверяем, что граф правильно загружен и имеет атрибут value для каждого узла
        G = nx.read_gml("datasets/football.gml")
        
        # Удостоверимся, что каждый узел имеет атрибут value
        for node in G.nodes():
            if "value" not in G.nodes[node]:
                if "conference" in G.nodes[node]:
                    # Преобразуем conference в числовое значение, если это строка
                    if isinstance(G.nodes[node]["conference"], str):
                        # Получим все уникальные значения конференций и сопоставим им числа
                        conferences = {node_data.get("conference") for _, node_data in G.nodes(data=True) 
                                    if "conference" in node_data}
                        conference_to_id = {conf: i for i, conf in enumerate(sorted(conferences))}
                        G.nodes[node]["value"] = conference_to_id[G.nodes[node]["conference"]]
                    else:
                        G.nodes[node]["value"] = G.nodes[node]["conference"]
                elif "gt" in G.nodes[node]:
                    # Если у графа есть атрибут gt (ground truth), используем его
                    G.nodes[node]["value"] = G.nodes[node]["gt"]
                else:
                    # Значение по умолчанию
                    G.nodes[node]["value"] = 0
        
        nx.write_gml(G, "datasets/football.gml")
        print("Processed football network")

def save_polbooks_network():
    """Download and save the political books network in GML format."""
    if not os.path.exists("datasets/polbooks.gml"):
        # Используем репозиторий vlivashkin/community-graphs
        download_from_github("gml_graphs", "polbooks.gml", "datasets/polbooks.gml")
        
        # Проверяем, что граф правильно загружен и имеет атрибут value для каждого узла
        G = nx.read_gml("datasets/polbooks.gml")
        
        # Удостоверимся, что каждый узел имеет атрибут value
        for node in G.nodes():
            if "value" not in G.nodes[node]:
                if "gt" in G.nodes[node]:
                    # Если у графа есть атрибут gt (ground truth), используем его
                    G.nodes[node]["value"] = G.nodes[node]["gt"]
                elif "color" in G.nodes[node]:
                    # Если у графа есть атрибут color, преобразуем его в значение
                    color_mapping = {"b": 0, "r": 1, "n": 2}  # blue, red, neutral
                    G.nodes[node]["value"] = color_mapping.get(G.nodes[node]["color"], 0)
                else:
                    # Значение по умолчанию
                    G.nodes[node]["value"] = 0
        
        nx.write_gml(G, "datasets/polbooks.gml")
        print("Processed political books network")

def save_dolphins_network():
    """Download and save the dolphins network in GML format."""
    if not os.path.exists("datasets/dolphins.gml"):
        # Используем репозиторий vlivashkin/community-graphs
        download_from_github("gml_graphs", "dolphins.gml", "datasets/dolphins.gml")
        
        # Проверяем, что граф правильно загружен и имеет атрибут value для каждого узла
        G = nx.read_gml("datasets/dolphins.gml")
        
        # Удостоверимся, что каждый узел имеет атрибут value
        for node in G.nodes():
            if "value" not in G.nodes[node]:
                if "gt" in G.nodes[node]:
                    # Если у графа есть атрибут gt (ground truth), используем его
                    G.nodes[node]["value"] = G.nodes[node]["gt"]
                else:
                    # Значение по умолчанию
                    G.nodes[node]["value"] = G.nodes[node].get("group", 0)
        
        nx.write_gml(G, "datasets/dolphins.gml")
        print("Processed dolphins network")

def save_polblogs_network():
    """Download and save the political blogs network in GML format."""
    if not os.path.exists("datasets/polblogs.gml"):
        # Используем репозиторий vlivashkin/community-graphs
        download_from_github("gml_graphs", "polblogs.gml", "datasets/polblogs.gml")
        
        # Проверяем, что граф правильно загружен и имеет атрибут value для каждого узла
        G = nx.read_gml("datasets/polblogs.gml")
        
        # Удостоверимся, что каждый узел имеет атрибут value
        for node in G.nodes():
            if "value" not in G.nodes[node]:
                if "gt" in G.nodes[node]:
                    # Если у графа есть атрибут gt (ground truth), используем его
                    G.nodes[node]["value"] = G.nodes[node]["gt"]
                elif "political" in G.nodes[node]:
                    # Преобразуем political в числовое значение: l -> 0, c -> 1
                    G.nodes[node]["value"] = 0 if G.nodes[node]["political"] == "l" else 1
                else:
                    # Значение по умолчанию
                    G.nodes[node]["value"] = 0
        
        nx.write_gml(G, "datasets/polblogs.gml")
        print("Processed political blogs network")

def save_cora_network():
    """Download and save the Cora citation network in GML format."""
    if not os.path.exists("datasets/cora.gml"):
        try:
            # Пробуем скачать из репозитория vlivashkin/community-graphs, но это большой файл,
            # поэтому может быть недоступен напрямую через raw.githubusercontent.com
            download_from_github("gml_graphs", "cora.gml", "datasets/cora.gml")
            
            # Проверяем, правильно ли загружен файл (если файл загружен напрямую, он может быть HTML с ошибкой)
            G = nx.read_gml("datasets/cora.gml")
            print("Processed Cora network from GitHub")
        except Exception as e:
            print(f"Failed to download Cora from GitHub: {e}")
            print("Trying alternative source...")
            
            # Альтернативный источник - LINQS
            url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
            cora_tgz = "datasets/cora.tgz"
            download_file(url, cora_tgz)
            
            # Создаем временную директорию для распаковки
            temp_dir = tempfile.mkdtemp()
            try:
                # Extract and convert to GML format
                with tarfile.open(cora_tgz, "r:gz") as tar:
                    tar.extractall(temp_dir)
                
                # Read cora content and citation files
                content_file = os.path.join(temp_dir, "cora", "cora.content")
                cites_file = os.path.join(temp_dir, "cora", "cora.cites")
                
                # Create NetworkX graph from Cora data
                G = nx.DiGraph()
                
                # Add nodes with attributes
                with open(content_file) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        node_id = int(parts[0])
                        features = parts[1:-1]
                        label = parts[-1]
                        # Преобразуем метку класса в числовое значение
                        class_mapping = {
                            "Case_Based": 0,
                            "Genetic_Algorithms": 1,
                            "Neural_Networks": 2,
                            "Probabilistic_Methods": 3,
                            "Reinforcement_Learning": 4,
                            "Rule_Learning": 5,
                            "Theory": 6
                        }
                        G.add_node(node_id, label=label, features=features, value=class_mapping.get(label, 0))
                
                # Add edges
                with open(cites_file) as f:
                    for line in f:
                        source, target = map(int, line.strip().split("\t"))
                        G.add_edge(source, target)
                
                # Save as GML
                nx.write_gml(G, "datasets/cora.gml")
                print("Converted Cora network to GML format")
            finally:
                # Clean up
                shutil.rmtree(temp_dir)
                os.remove(cora_tgz)

def save_facebook_network():
    """Download and save the Facebook network in GML format."""
    if not os.path.exists("datasets/facebook.gml"):
        # Используем SNAP Stanford repository для Facebook
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        facebook_gz = "datasets/facebook_combined.txt.gz"
        download_file(url, facebook_gz)
        
        # Создаем граф из edgelist файла
        G = nx.read_edgelist(facebook_gz, nodetype=int, create_using=nx.Graph())
        
        # Создаем фиктивные community для бенчмарка
        # Используем простую кластеризацию на основе connected components или label propagation
        try:
            # Попробуем использовать встроенный алгоритм label propagation
            communities = list(nx.algorithms.community.label_propagation_communities(G))
            # Преобразуем список сообществ в словарь {node: community_id}
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i
            nx.set_node_attributes(G, node_to_community, "value")
        except (ImportError, AttributeError):
            # Если не удалось, используем простое присвоение
            for node in G.nodes():
                G.nodes[node]["value"] = node % 10  # Простая группировка по модулю 10
        
        nx.write_gml(G, "datasets/facebook.gml")
        os.remove(facebook_gz)
        print("Converted Facebook network to GML format")

def save_citeseer_network():
    """Download and save the CiteSeer network in GML format."""
    output_path = "datasets/citeseer.gml"
    if not os.path.exists(output_path):
        # Используем репозиторий vlivashkin/community-graphs
        download_from_github("gml_graphs", "citeseer.gml", output_path)
        
        # Проверяем, что граф правильно загружен и имеет атрибут value
        G = nx.read_gml(output_path)
        
        # Удостоверимся, что каждый узел имеет атрибут value из gt
        for node in G.nodes():
            if "value" not in G.nodes[node]:
                if "gt" in G.nodes[node]:
                    G.nodes[node]["value"] = G.nodes[node]["gt"]
                else:
                    # Значение по умолчанию, если gt отсутствует
                    G.nodes[node]["value"] = 0 
        
        nx.write_gml(G, output_path)
        print("Processed CiteSeer network")
    else:
        print(f"{output_path} already exists")

def save_email_eu_core_network():
    """Download and save the Email EU Core network in GML format."""
    output_path = "datasets/email_eu_core.gml"
    if not os.path.exists(output_path):
        # Используем репозиторий vlivashkin/community-graphs
        download_from_github("gml_graphs", "eu-core.gml", output_path)
        
        # Проверяем, что граф правильно загружен и имеет атрибут value
        G = nx.read_gml(output_path)
        
        # Удостоверимся, что каждый узел имеет атрибут value из gt
        for node in G.nodes():
            if "value" not in G.nodes[node]:
                if "gt" in G.nodes[node]:
                    G.nodes[node]["value"] = G.nodes[node]["gt"]
                else:
                     # Значение по умолчанию, если gt отсутствует
                    G.nodes[node]["value"] = 0
        
        nx.write_gml(G, output_path)
        print("Processed Email EU Core network")
    else:
        print(f"{output_path} already exists")

def save_orkut_network():
    """Download Orkut dataset files (edge list and communities) from SNAP."""
    print("\nProcessing Orkut dataset...")
    # Edge list
    download_file(
        "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        "datasets/com-orkut.ungraph.txt.gz"
    )
    # Communities
    download_file(
        "https://snap.stanford.edu/data/bigdata/communities/com-orkut.all.cmty.txt.gz",
        "datasets/com-orkut.all.cmty.txt.gz"
    )
    # Note: These files need to be unzipped manually or by the loader function later
    print("Orkut dataset files downloaded (remember to unzip them: gunzip datasets/com-orkut.*.gz)")


def save_livejournal_network():
    """Download LiveJournal dataset files (edge list and communities) from SNAP."""
    print("\nProcessing LiveJournal dataset...")
    # Edge list
    download_file(
        "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        "datasets/com-lj.ungraph.txt.gz"
    )
    # Communities
    download_file(
        "https://snap.stanford.edu/data/bigdata/communities/com-lj.all.cmty.txt.gz",
        "datasets/com-lj.all.cmty.txt.gz"
    )
    # Note: These files need to be unzipped manually or by the loader function later
    print("LiveJournal dataset files downloaded (remember to unzip them: gunzip datasets/com-lj.*.gz)")

def main():
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    
    save_football_network()
    save_polbooks_network()
    save_dolphins_network()
    save_polblogs_network()
    # save_cora_network()
    save_facebook_network()
    save_citeseer_network()  # Добавляем вызов для Citeseer
    save_email_eu_core_network() # Добавляем вызов для Email-Eu-core

    # Add downloads for large SNAP datasets
    save_orkut_network()
    save_livejournal_network()

if __name__ == "__main__":
    main()
