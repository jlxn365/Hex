#!/usr/bin/env python3
"""
Script pour patcher automatiquement votre player.py avec les métriques.
Usage: python patch_player.py
"""

import os
import shutil
from datetime import datetime

def backup_file(filename):
    """Crée une sauvegarde du fichier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{filename}.backup_{timestamp}"
    shutil.copy2(filename, backup_name)
    print(f"✅ Sauvegarde créée: {backup_name}")
    return backup_name

def patch_player_file():
    """Patch le fichier player.py avec les métriques."""
    
    filename = "player.py"
    
    if not os.path.exists(filename):
        print(f"❌ Fichier {filename} non trouvé!")
        return False
    
    print("\n" + "="*70)
    print(" "*20 + "PATCHING DE PLAYER.PY")
    print("="*70 + "\n")
    
    # Créer une sauvegarde
    backup_file(filename)
    
    # Lire le fichier actuel
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier si déjà patché
    if '_performance_data' in content and 'save_performance_data' in content:
        print("⚠️  Le fichier semble déjà patché!")
        response = input("Voulez-vous forcer le patch? (o/n): ")
        if response.lower() != 'o':
            print("Annulé.")
            return False
    
    # Ajouter les imports nécessaires
    imports_to_add = """
import time
import json
"""
    
    if 'import time' not in content:
        # Trouver où ajouter les imports (après les autres imports)
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ')):
                import_lines.append(line)
            else:
                if line.strip() and in_imports and not line.startswith('#'):
                    in_imports = False
                other_lines.append(line)
        
        # Reconstruire avec les nouveaux imports
        import_lines.append(imports_to_add)
        content = '\n'.join(import_lines) + '\n' + '\n'.join(other_lines)
        print("✅ Imports ajoutés")
    
    # Trouver la classe MyPlayer
    class_start = content.find('class MyPlayer(')
    if class_start == -1:
        print("❌ Classe MyPlayer non trouvée!")
        return False
    
    # Trouver le __init__
    init_start = content.find('def __init__(', class_start)
    if init_start == -1:
        print("❌ Méthode __init__ non trouvée!")
        return False
    
    # Ajouter la variable de classe AVANT __init__
    performance_data_code = '''
    # Variable de classe pour le tracking global
    _performance_data = {
        'execution_times': [],
        'move_numbers': [],
        'board_fills': [],
        'search_depths': [],
        'nodes_explored': [],
        'tt_hits': [],
        'tt_misses': [],
        'alpha_cutoffs': [],
        'my_distances': [],
        'opp_distances': [],
        'branching_factors': [],
        'evaluation_scores': [],
        'winning_moves': 0,
        'blocking_moves': 0,
        'game_start_time': None
    }
'''
    
    if '_performance_data' not in content:
        # Insérer avant def __init__
        content = content[:init_start] + performance_data_code + '\n    ' + content[init_start:]
        print("✅ Variable _performance_data ajoutée")
    
    # Ajouter les attributs dans __init__
    init_attrs = '''
        
        # Compteurs pour métriques
        self.nodes_explored = 0
        self.tt_hits = 0
        self.tt_misses = 0
        self.alpha_cutoffs = 0
        self.move_count = 0
        
        # Initialiser le temps de début si première instance
        if MyPlayer._performance_data['game_start_time'] is None:
            MyPlayer._performance_data['game_start_time'] = time.time()
'''
    
    if 'self.nodes_explored' not in content:
        # Trouver la fin de __init__ (chercher self.dijkstra_cache = {})
        dijkstra_pos = content.find('self.dijkstra_cache = {}')
        if dijkstra_pos != -1:
            # Insérer après
            eol = content.find('\n', dijkstra_pos)
            content = content[:eol] + init_attrs + content[eol:]
            print("✅ Attributs de tracking ajoutés dans __init__")
        else:
            print("⚠️  Position d'insertion non trouvée, veuillez ajouter manuellement")
    
    # Ajouter les méthodes à la fin de la classe
    methods_code = '''
    
    def _record_move_metrics(self, start_time, board, depth, branching_factor, 
                            board_fill, winning_found, blocking_found, eval_score):
        """Enregistre les métriques du coup actuel."""
        execution_time = time.time() - start_time
        
        my_dist = self._shortest_path_dijkstra(board, self.piece_type)
        opp_dist = self._shortest_path_dijkstra(board, self._opponent())
        
        # Ajouter aux données globales
        MyPlayer._performance_data['execution_times'].append(execution_time)
        MyPlayer._performance_data['move_numbers'].append(self.move_count)
        MyPlayer._performance_data['board_fills'].append(board_fill)
        MyPlayer._performance_data['search_depths'].append(depth)
        MyPlayer._performance_data['nodes_explored'].append(self.nodes_explored)
        MyPlayer._performance_data['tt_hits'].append(self.tt_hits)
        MyPlayer._performance_data['tt_misses'].append(self.tt_misses)
        MyPlayer._performance_data['alpha_cutoffs'].append(self.alpha_cutoffs)
        MyPlayer._performance_data['my_distances'].append(my_dist)
        MyPlayer._performance_data['opp_distances'].append(opp_dist)
        MyPlayer._performance_data['branching_factors'].append(branching_factor)
        MyPlayer._performance_data['evaluation_scores'].append(eval_score)
    
    @classmethod
    def save_performance_data(cls, filename='performance_data.json'):
        """Sauvegarde les données de performance en JSON."""
        if cls._performance_data['game_start_time']:
            cls._performance_data['total_game_time'] = time.time() - cls._performance_data['game_start_time']
        
        with open(filename, 'w') as f:
            json.dump(cls._performance_data, f, indent=2)
        
        print(f"\\n✅ Données de performance sauvegardées dans {filename}")
    
    @classmethod
    def reset_performance_data(cls):
        """Réinitialise les données de performance."""
        cls._performance_data = {
            'execution_times': [],
            'move_numbers': [],
            'board_fills': [],
            'search_depths': [],
            'nodes_explored': [],
            'tt_hits': [],
            'tt_misses': [],
            'alpha_cutoffs': [],
            'my_distances': [],
            'opp_distances': [],
            'branching_factors': [],
            'evaluation_scores': [],
            'winning_moves': 0,
            'blocking_moves': 0,
            'game_start_time': time.time()
        }
'''
    
    if 'def _record_move_metrics' not in content:
        # Ajouter à la fin de la classe (avant la dernière ligne ou à la fin du fichier)
        content = content.rstrip() + '\n' + methods_code + '\n'
        print("✅ Méthodes de performance ajoutées")
    
    # Sauvegarder le fichier modifié
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*70)
    print("✅ PATCHING TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print("\n⚠️  IMPORTANT: Vous devez encore modifier manuellement:")
    print("\n1. Dans compute_action(), ajoutez au DÉBUT:")
    print("   start_time = time.time()")
    print("   self.nodes_explored = 0")
    print("   self.tt_hits = 0")
    print("   self.tt_misses = 0")
    print("   self.alpha_cutoffs = 0")
    print("   self.move_count += 1")
    print("   board_fill_percentage = (len(env) / (dim * dim)) * 100")
    print("\n2. Dans compute_action(), avant CHAQUE return, ajoutez:")
    print("   self._record_move_metrics(start_time, board, depth, len(actions),")
    print("                            board_fill_percentage, winning_found, blocking_found, score)")
    print("\n3. Dans _minimax(), ajoutez:")
    print("   - Au début: self.nodes_explored += 1")
    print("   - Dans TT lookup: self.tt_hits += 1 ou self.tt_misses += 1")
    print("   - Lors des coupures: self.alpha_cutoffs += 1")
    print("\n4. Exécutez: python verify_setup.py pour vérifier")
    print()
    
    return True

if __name__ == "__main__":
    patch_player_file()