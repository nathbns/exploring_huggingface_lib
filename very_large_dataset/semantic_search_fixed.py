#!/usr/bin/env python3
"""
Moteur de recherche sémantique avec 🤗 Datasets et FAISS
Solution au problème DatasetGenerationError

Basé sur le tutoriel Hugging Face : https://huggingface.co/learn/llm-course/fr/chapter5/5
"""

import requests
import time
import math
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset

def fetch_issues_clean(
    owner="huggingface",
    repo="datasets",
    num_issues=1000,  # Réduisons pour éviter les problèmes
    issues_path=Path("."),
):
    """
    Récupère les issues GitHub et les nettoie pour éviter les problèmes de typage.
    
    Le problème principal était que l'API GitHub retourne des données avec :
    - Des valeurs null qui causent des problèmes de typage
    - Des champs complexes imbriqués
    - Des types inconsistants entre les issues
    """
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    all_issues = []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"
    headers = {}  # Pas de token pour ce test

    print(f"Récupération de {num_issues} issues sur {num_pages} pages...")
    
    for page in tqdm(range(1, num_pages + 1)):
        query = f"issues?page={page}&per_page={per_page}&state=all"
        
        try:
            response = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
            response.raise_for_status()
            
            page_issues = response.json()
            
            if not page_issues:
                print(f"Aucune issue trouvée à la page {page}")
                break
                
            # SOLUTION : Nettoyer et simplifier les données pour éviter les problèmes de typage
            cleaned_issues = []
            for issue in page_issues:
                # Ne garder que les champs essentiels avec des types cohérents
                cleaned_issue = {
                    'url': issue.get('html_url', ''),
                    'id': issue.get('id', 0),
                    'number': issue.get('number', 0),
                    'title': issue.get('title', ''),
                    'state': issue.get('state', ''),
                    'body': issue.get('body') or '',  # IMPORTANT: Remplacer None par chaîne vide
                    'user': issue.get('user', {}).get('login', ''),
                    'labels': [label.get('name', '') for label in issue.get('labels', [])],
                    'comments': issue.get('comments', 0),
                    'created_at': issue.get('created_at', ''),
                    'updated_at': issue.get('updated_at', ''),
                    'is_pull_request': issue.get('pull_request') is not None  # Booléen simple
                }
                cleaned_issues.append(cleaned_issue)
            
            all_issues.extend(cleaned_issues)
            
            # Pause pour respecter les limites de l'API
            time.sleep(0.5)
            
            if len(all_issues) >= num_issues:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête page {page}: {e}")
            break

    # Limiter au nombre demandé
    all_issues = all_issues[:num_issues]
    
    # Sauvegarder avec un nom différent pour éviter les conflits
    output_file = issues_path / f"{repo}-issues-cleaned.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for issue in all_issues:
            json.dump(issue, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ {len(all_issues)} issues sauvegardées dans {output_file}")
    return output_file

def clean_issues(dataset):
    """Nettoie le dataset des issues GitHub selon le tutoriel."""
    print(f"Dataset initial: {len(dataset)} issues")
    
    # Filtrer les pull requests
    dataset = dataset.filter(lambda x: not x["is_pull_request"])
    print(f"Après filtrage des PR: {len(dataset)} issues")
    
    # Filtrer les issues sans corps ou trop courtes
    dataset = dataset.filter(lambda x: x["body"] is not None and len(x["body"]) > 15)
    print(f"Après filtrage des issues vides: {len(dataset)} issues")
    
    # Créer une colonne de texte combiné
    def create_text(example):
        title = example["title"] if example["title"] else ""
        body = example["body"] if example["body"] else ""
        return {"text": f"{title}\n\n{body}"}
    
    dataset = dataset.map(create_text)
    print(f"Texte combiné créé pour {len(dataset)} issues")
    
    return dataset

def main():
    """Fonction principale pour tester la solution."""
    print("🔧 Solution au problème DatasetGenerationError")
    print("=" * 50)
    
    # 1. Récupérer et nettoyer les données
    print("\n1. Récupération des issues...")
    issues_file = fetch_issues_clean()
    
    # 2. Charger le dataset
    print("\n2. Chargement du dataset...")
    try:
        issues_dataset = load_dataset('json', data_files=str(issues_file), split='train')
        print(f"✅ Dataset chargé avec succès!")
        print(f"Nombre d'issues: {len(issues_dataset)}")
        print(f"Colonnes: {issues_dataset.column_names}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # 3. Examiner un échantillon
    print("\n3. Examen d'un échantillon...")
    if len(issues_dataset) > 0:
        sample = issues_dataset[0]
        print("Première issue:")
        print(f"Titre: {sample['title']}")
        print(f"État: {sample['state']}")
        print(f"Utilisateur: {sample['user']}")
        print(f"Est une PR: {sample['is_pull_request']}")
        print(f"Corps (premiers 200 caractères): {sample['body'][:200]}...")
    
    # 4. Nettoyer selon le tutoriel
    print("\n4. Nettoyage des données...")
    issues_dataset = clean_issues(issues_dataset)
    
    # 5. Examiner le résultat final
    print("\n5. Résultat final...")
    if len(issues_dataset) > 0:
        sample = issues_dataset[0]
        print("Texte de la première issue (premiers 300 caractères):")
        print(sample['text'][:300] + "...")
    
    print(f"\n✅ Succès! Dataset final: {len(issues_dataset)} issues prêtes pour la recherche sémantique")

if __name__ == "__main__":
    main()
