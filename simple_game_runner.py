#!/usr/bin/env python3
"""
Script simple pour lancer une partie et sauvegarder les métriques.
Usage: python simple_game_runner.py
"""

import sys
import os
from os.path import basename, splitext, dirname

# Importer vos modules
from board_hex import BoardHex
from player_hex import PlayerHex
from master_hex import MasterHex
from game_state_hex import GameStateHex
from seahorse.utils.gui_client import GUIClient
from seahorse.utils.custom_exceptions import PlayerDuplicateError

def play_game_with_metrics(player1_path="player.py", player2_path="player.py", 
                           use_gui=True, save_metrics=True):
    """
    Lance une partie et sauvegarde automatiquement les métriques.
    
    Args:
        player1_path: Chemin vers le premier joueur
        player2_path: Chemin vers le deuxième joueur
        use_gui: Afficher l'interface graphique
        save_metrics: Sauvegarder les métriques après la partie
    """
    
    print("\n" + "="*70)
    print(" "*20 + "LANCEMENT DE LA PARTIE")
    print("="*70 + "\n")
    
    # Charger les joueurs
    try:
        folder1 = dirname(player1_path) if dirname(player1_path) else "."
        sys.path.insert(0, folder1)
        player1_module = __import__(splitext(basename(player1_path))[0], fromlist=[None])
        
        folder2 = dirname(player2_path) if dirname(player2_path) else "."
        sys.path.insert(0, folder2)
        player2_module = __import__(splitext(basename(player2_path))[0], fromlist=[None])
        
        # Créer les instances
        player1 = player1_module.MyPlayer("R", name="Player1")
        player2 = player2_module.MyPlayer("B", name="Player2")
        
        print(f"✅ Joueur 1 chargé: {player1.get_name()}")
        print(f"✅ Joueur 2 chargé: {player2.get_name()}\n")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des joueurs: {e}")
        return False
    
    # Initialiser le jeu
    time_limit = 60 * 15
    list_players = [player1, player2]
    init_scores = {player1.get_id(): 0, player2.get_id(): 0}
    dim = [14, 14]
    env = {}
    
    init_rep = BoardHex(env=env, dim=dim)
    initial_game_state = GameStateHex(
        scores=init_scores, 
        next_player=player1, 
        players=list_players, 
        rep=init_rep, 
        step=0
    )
    
    try:
        master = MasterHex(
            name="Hex",
            initial_game_state=initial_game_state,
            players_iterator=list_players,
            log_level="INFO",
            port=16001,
            hostname="localhost",
            time_limit=time_limit
        )
    except PlayerDuplicateError:
        print("❌ Erreur: Joueurs dupliqués")
        return False
    
    # Lancer le jeu avec ou sans GUI
    listeners = []
    if use_gui:
        gui_path = os.path.join(dirname(os.path.abspath(__file__)), 'GUI', 'index.html')
        if os.path.exists(gui_path):
            listeners.append(GUIClient(path=gui_path))
        else:
            print("⚠️  GUI non trouvée, lancement en mode headless")
    
    print("🎮 La partie commence...\n")
    
    try:
        master.record_game(listeners=listeners)
        print("\n✅ Partie terminée!\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Partie interrompue par l'utilisateur\n")
        return False
    except Exception as e:
        print(f"\n❌ Erreur pendant la partie: {e}\n")
        return False
    
    # Sauvegarder les métriques
    if save_metrics:
        print("💾 Sauvegarde des métriques de performance...\n")
        
        try:
            # Importer directement depuis le module chargé
            player1_module.MyPlayer.save_performance_data('performance_data.json')
            print("✅ Métriques sauvegardées dans performance_data.json\n")
            
            # Afficher un résumé rapide
            import json
            with open('performance_data.json', 'r') as f:
                data = json.load(f)
                
            if data['execution_times']:
                import numpy as np
                avg_time = np.mean(data['execution_times'])
                total_moves = len(data['execution_times'])
                
                print("📊 Résumé rapide:")
                print(f"  • Nombre de coups: {total_moves}")
                print(f"  • Temps moyen/coup: {avg_time:.4f}s")
                print(f"  • Temps total: {sum(data['execution_times']):.2f}s")
                
                if data.get('winning_moves', 0) > 0:
                    print(f"  • Coups gagnants détectés: {data['winning_moves']}")
                if data.get('blocking_moves', 0) > 0:
                    print(f"  • Coups bloquants détectés: {data['blocking_moves']}")
                
                print("\n📈 Pour générer les graphiques et le rapport:")
                print("  python analyze_performance.py performance_data.json\n")
            
            return True
            
        except AttributeError:
            print("❌ La méthode save_performance_data() n'existe pas dans MyPlayer")
            print("   Assurez-vous d'avoir modifié votre player.py correctement.\n")
            return False
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde des métriques: {e}\n")
            return False
    
    return True


def main():
    """Point d'entrée principal."""
    
    # Configuration
    player1_path = "player.py"         # Votre IA avec métriques
    player2_path = "random_player_hex.py"  # Joueur aléatoire
    use_gui = True                     # False pour mode headless
    
    # Lancer la partie
    success = play_game_with_metrics(
        player1_path=player1_path,
        player2_path=player2_path,
        use_gui=use_gui,
        save_metrics=True
    )
    
    if success:
        print("="*70)
        print("✅ SUCCÈS!")
        print("="*70 + "\n")
    else:
        print("="*70)
        print("❌ ÉCHEC")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    # Gérer les arguments en ligne de commande (optionnel)
    import argparse
    
    parser = argparse.ArgumentParser(description="Lance une partie de Hex avec métriques")
    parser.add_argument("--player1", default="player.py", help="Chemin vers le joueur 1")
    parser.add_argument("--player2", default="player.py", help="Chemin vers le joueur 2")
    parser.add_argument("--no-gui", action="store_true", help="Désactiver l'interface graphique")
    parser.add_argument("--no-metrics", action="store_true", help="Ne pas sauvegarder les métriques")
    
    args = parser.parse_args()
    
    success = play_game_with_metrics(
        player1_path=args.player1,
        player2_path=args.player2,
        use_gui=not args.no_gui,
        save_metrics=not args.no_metrics
    )
    
    sys.exit(0 if success else 1)
