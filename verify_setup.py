#!/usr/bin/env python3
"""
Script de vérification pour s'assurer que tout est correctement configuré.
Usage: python verify_setup.py
"""

import sys
import os

def verify_setup():
    """Vérifie que tout est configuré correctement."""
    
    print("\n" + "="*70)
    print(" "*15 + "VÉRIFICATION DE LA CONFIGURATION")
    print("="*70 + "\n")
    
    all_ok = True
    
    # 1. Vérifier les dépendances
    print("📦 Vérification des dépendances...\n")
    
    try:
        import matplotlib
        print(f"  ✅ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  ❌ matplotlib manquant → pip install matplotlib")
        all_ok = False
    
    try:
        import numpy
        print(f"  ✅ numpy {numpy.__version__}")
    except ImportError:
        print("  ❌ numpy manquant → pip install numpy")
        all_ok = False
    
    print()
    
    # 2. Vérifier les fichiers
    print("📁 Vérification des fichiers...\n")
    
    required_files = {
        'player.py': 'Votre agent',
        'analyze_performance.py': "Script d'analyse",
        'main_hex.py': 'Script de jeu'
    }
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"  ✅ {filename} - {description}")
        else:
            print(f"  ❌ {filename} manquant - {description}")
            all_ok = False
    
    print()
    
    # 3. Vérifier player.py
    print("🔍 Vérification de player.py...\n")
    
    try:
        # Importer le module
        import player
        
        # Vérifier que MyPlayer existe
        if hasattr(player, 'MyPlayer'):
            print("  ✅ Classe MyPlayer trouvée")
            
            # Vérifier les méthodes
            required_methods = [
                'save_performance_data',
                'reset_performance_data',
                '_record_move_metrics'
            ]
            
            for method in required_methods:
                if hasattr(player.MyPlayer, method):
                    print(f"  ✅ Méthode {method}() trouvée")
                else:
                    print(f"  ❌ Méthode {method}() manquante")
                    all_ok = False
            
            # Vérifier la variable de classe
            if hasattr(player.MyPlayer, '_performance_data'):
                print("  ✅ Variable _performance_data trouvée")
            else:
                print("  ❌ Variable _performance_data manquante")
                all_ok = False
            
            # Vérifier les attributs d'instance
            try:
                test_player = player.MyPlayer("R", name="Test")
                
                required_attrs = [
                    'nodes_explored',
                    'tt_hits',
                    'tt_misses',
                    'alpha_cutoffs',
                    'move_count'
                ]
                
                for attr in required_attrs:
                    if hasattr(test_player, attr):
                        print(f"  ✅ Attribut {attr} trouvé")
                    else:
                        print(f"  ❌ Attribut {attr} manquant")
                        all_ok = False
                
            except Exception as e:
                print(f"  ⚠️  Impossible de créer une instance: {e}")
                all_ok = False
        else:
            print("  ❌ Classe MyPlayer non trouvée dans player.py")
            all_ok = False
            
    except ImportError as e:
        print(f"  ❌ Impossible d'importer player.py: {e}")
        all_ok = False
    except Exception as e:
        print(f"  ❌ Erreur lors de la vérification: {e}")
        all_ok = False
    
    print()
    
    # 4. Test fonctionnel rapide
    print("🧪 Test fonctionnel...\n")
    
    try:
        import player
        
        # Réinitialiser les données
        player.MyPlayer.reset_performance_data()
        print("  ✅ reset_performance_data() fonctionne")
        
        # Simuler quelques données
        player.MyPlayer._performance_data['execution_times'].append(0.1)
        player.MyPlayer._performance_data['move_numbers'].append(1)
        player.MyPlayer._performance_data['board_fills'].append(5.0)
        
        # Tester la sauvegarde
        test_file = 'test_metrics.json'
        player.MyPlayer.save_performance_data(test_file)
        
        if os.path.exists(test_file):
            print(f"  ✅ save_performance_data() fonctionne")
            
            # Vérifier le contenu
            import json
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            if 'execution_times' in data and len(data['execution_times']) > 0:
                print("  ✅ Données correctement sauvegardées")
            else:
                print("  ❌ Données manquantes dans le JSON")
                all_ok = False
            
            # Nettoyer
            os.remove(test_file)
            print("  ✅ Test nettoyé")
        else:
            print(f"  ❌ Fichier {test_file} non créé")
            all_ok = False
            
    except Exception as e:
        print(f"  ❌ Erreur lors du test: {e}")
        all_ok = False
    
    print()
    
    # Résultat final
    print("="*70)
    if all_ok:
        print("✅ TOUT EST CONFIGURÉ CORRECTEMENT!")
        print("="*70)
        print("\nVous pouvez maintenant:")
        print("  1. Lancer une partie: python simple_game_runner.py")
        print("  2. Analyser les résultats: python analyze_performance.py performance_data.json")
        print()
        return True
    else:
        print("❌ CONFIGURATION INCOMPLÈTE")
        print("="*70)
        print("\n⚠️  Veuillez corriger les problèmes ci-dessus avant de continuer.\n")
        print("📚 Consultez le guide d'intégration pour plus d'aide.")
        print()
        return False


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)