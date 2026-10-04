import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

class PerformanceTracker:
    """
    Classe pour suivre et analyser les performances de l'agent Hex.
    """
    
    def __init__(self):
        self.metrics = {
            'execution_times': [],  # Temps par coup
            'move_numbers': [],     # Numéro du coup
            'board_fills': [],      # % du plateau rempli
            'search_depths': [],    # Profondeur de recherche utilisée
            'nodes_explored': [],   # Nœuds explorés (minimax)
            'tt_hits': [],          # Hits du transposition table
            'tt_misses': [],        # Misses du transposition table
            'alpha_cutoffs': [],    # Nombre de coupures alpha-beta
            'my_distances': [],     # Distance shortest path (nous)
            'opp_distances': [],    # Distance shortest path (adversaire)
            'branching_factors': [], # Facteur de branchement effectif
            'evaluation_scores': [], # Scores d'évaluation
            'cache_sizes': [],      # Taille des caches
            'winning_moves_found': [], # Coups gagnants détectés
            'blocking_moves_found': [], # Coups bloquants détectés
        }
        
        self.game_results = {
            'wins': 0,
            'losses': 0,
            'total_time': 0,
            'average_time_per_move': 0,
            'max_time': 0,
            'min_time': float('inf')
        }
    
    def record_move(self, **kwargs):
        """Enregistre les métriques d'un coup."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def update_game_result(self, won: bool, total_time: float):
        """Met à jour les résultats de la partie."""
        if won:
            self.game_results['wins'] += 1
        else:
            self.game_results['losses'] += 1
        
        self.game_results['total_time'] += total_time
        
        if self.metrics['execution_times']:
            avg_time = np.mean(self.metrics['execution_times'])
            self.game_results['average_time_per_move'] = avg_time
            self.game_results['max_time'] = max(self.metrics['execution_times'])
            self.game_results['min_time'] = min(self.metrics['execution_times'])
    
    def plot_execution_time_vs_game_progress(self, save_path='execution_time_analysis.png'):
        """Graphique: Temps d'exécution en fonction de l'avancement."""
        if not self.metrics['execution_times']:
            print("Aucune donnée à afficher")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analyse du Temps d\'Exécution', fontsize=16, fontweight='bold')
        
        # 1. Temps vs Numéro de coup
        ax1 = axes[0, 0]
        ax1.plot(self.metrics['move_numbers'], self.metrics['execution_times'], 
                marker='o', markersize=4, linewidth=1, alpha=0.7)
        ax1.set_xlabel('Numéro du Coup')
        ax1.set_ylabel('Temps (secondes)')
        ax1.set_title('Temps d\'Exécution par Coup')
        ax1.grid(True, alpha=0.3)
        
        # Ajouter une ligne de tendance
        z = np.polyfit(self.metrics['move_numbers'], self.metrics['execution_times'], 2)
        p = np.poly1d(z)
        ax1.plot(self.metrics['move_numbers'], p(self.metrics['move_numbers']), 
                "r--", alpha=0.5, label='Tendance')
        ax1.legend()
        
        # 2. Temps vs % Plateau Rempli
        ax2 = axes[0, 1]
        if self.metrics['board_fills']:
            ax2.scatter(self.metrics['board_fills'], self.metrics['execution_times'], 
                       alpha=0.6, c=self.metrics['move_numbers'], cmap='viridis')
            ax2.set_xlabel('Plateau Rempli (%)')
            ax2.set_ylabel('Temps (secondes)')
            ax2.set_title('Temps vs Complexité du Plateau')
            ax2.grid(True, alpha=0.3)
            cbar = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar.set_label('Numéro du Coup')
        
        # 3. Distribution des temps
        ax3 = axes[1, 0]
        ax3.hist(self.metrics['execution_times'], bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(self.metrics['execution_times']), 
                   color='r', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(self.metrics["execution_times"]):.3f}s')
        ax3.axvline(np.median(self.metrics['execution_times']), 
                   color='g', linestyle='--', linewidth=2, label=f'Médiane: {np.median(self.metrics["execution_times"]):.3f}s')
        ax3.set_xlabel('Temps (secondes)')
        ax3.set_ylabel('Fréquence')
        ax3.set_title('Distribution des Temps d\'Exécution')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Temps cumulé
        ax4 = axes[1, 1]
        cumulative_time = np.cumsum(self.metrics['execution_times'])
        ax4.plot(self.metrics['move_numbers'], cumulative_time, linewidth=2)
        ax4.set_xlabel('Numéro du Coup')
        ax4.set_ylabel('Temps Cumulé (secondes)')
        ax4.set_title('Temps Cumulé au Cours de la Partie')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(self.metrics['move_numbers'], cumulative_time, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_search_efficiency(self, save_path='search_efficiency.png'):
        """Graphique: Efficacité de la recherche."""
        if not self.metrics['nodes_explored']:
            print("Aucune donnée de recherche disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Efficacité de la Recherche Minimax', fontsize=16, fontweight='bold')
        
        # 1. Nœuds explorés vs temps
        ax1 = axes[0, 0]
        ax1.scatter(self.metrics['nodes_explored'], self.metrics['execution_times'], alpha=0.6)
        ax1.set_xlabel('Nœuds Explorés')
        ax1.set_ylabel('Temps (secondes)')
        ax1.set_title('Temps vs Nœuds Explorés')
        ax1.grid(True, alpha=0.3)
        
        # 2. Taux de hit du TT
        ax2 = axes[0, 1]
        if self.metrics['tt_hits'] and self.metrics['tt_misses']:
            tt_total = np.array(self.metrics['tt_hits']) + np.array(self.metrics['tt_misses'])
            tt_rate = np.array(self.metrics['tt_hits']) / tt_total * 100
            ax2.plot(self.metrics['move_numbers'], tt_rate, marker='o', markersize=4)
            ax2.set_xlabel('Numéro du Coup')
            ax2.set_ylabel('Taux de Hit TT (%)')
            ax2.set_title('Efficacité du Transposition Table')
            ax2.axhline(np.mean(tt_rate), color='r', linestyle='--', 
                       label=f'Moyenne: {np.mean(tt_rate):.1f}%')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Profondeur de recherche
        ax3 = axes[1, 0]
        if self.metrics['search_depths']:
            ax3.plot(self.metrics['move_numbers'], self.metrics['search_depths'], 
                    marker='s', markersize=5, linewidth=2)
            ax3.set_xlabel('Numéro du Coup')
            ax3.set_ylabel('Profondeur')
            ax3.set_title('Profondeur de Recherche Adaptative')
            ax3.grid(True, alpha=0.3)
            ax3.set_yticks(range(int(min(self.metrics['search_depths'])), 
                                int(max(self.metrics['search_depths']))+1))
        
        # 4. Coupures Alpha-Beta
        ax4 = axes[1, 1]
        if self.metrics['alpha_cutoffs']:
            ax4.bar(self.metrics['move_numbers'], self.metrics['alpha_cutoffs'], alpha=0.7)
            ax4.set_xlabel('Numéro du Coup')
            ax4.set_ylabel('Nombre de Coupures')
            ax4.set_title('Efficacité de l\'Élagage Alpha-Beta')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_strategic_metrics(self, save_path='strategic_metrics.png'):
        """Graphique: Métriques stratégiques."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analyse Stratégique', fontsize=16, fontweight='bold')
        
        # 1. Distances shortest path
        ax1 = axes[0, 0]
        if self.metrics['my_distances'] and self.metrics['opp_distances']:
            ax1.plot(self.metrics['move_numbers'], self.metrics['my_distances'], 
                    label='Notre Distance', marker='o', linewidth=2)
            ax1.plot(self.metrics['move_numbers'], self.metrics['opp_distances'], 
                    label='Distance Adversaire', marker='s', linewidth=2)
            ax1.set_xlabel('Numéro du Coup')
            ax1.set_ylabel('Distance')
            ax1.set_title('Évolution des Distances Shortest Path')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Avantage stratégique
        ax2 = axes[0, 1]
        if self.metrics['my_distances'] and self.metrics['opp_distances']:
            advantage = np.array(self.metrics['opp_distances']) - np.array(self.metrics['my_distances'])
            colors = ['green' if a > 0 else 'red' for a in advantage]
            ax2.bar(self.metrics['move_numbers'], advantage, color=colors, alpha=0.7)
            ax2.axhline(0, color='black', linewidth=1)
            ax2.set_xlabel('Numéro du Coup')
            ax2.set_ylabel('Avantage (Distance Opp - Notre Distance)')
            ax2.set_title('Avantage Stratégique')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Scores d'évaluation
        ax3 = axes[1, 0]
        if self.metrics['evaluation_scores']:
            ax3.plot(self.metrics['move_numbers'], self.metrics['evaluation_scores'], 
                    linewidth=2, color='purple')
            ax3.set_xlabel('Numéro du Coup')
            ax3.set_ylabel('Score d\'Évaluation')
            ax3.set_title('Évolution du Score d\'Évaluation')
            ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)
        
        # 4. Facteur de branchement
        ax4 = axes[1, 1]
        if self.metrics['branching_factors']:
            ax4.plot(self.metrics['move_numbers'], self.metrics['branching_factors'], 
                    marker='o', linewidth=2, color='orange')
            ax4.set_xlabel('Numéro du Coup')
            ax4.set_ylabel('Facteur de Branchement')
            ax4.set_title('Complexité de l\'Arbre de Recherche')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
        plt.close()
    
    def generate_summary_report(self, save_path='performance_report.txt'):
        """Génère un rapport textuel détaillé."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT DE PERFORMANCE - AGENT HEX\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Résultats généraux
            f.write("1. RÉSULTATS GÉNÉRAUX\n")
            f.write("-" * 60 + "\n")
            f.write(f"Victoires: {self.game_results['wins']}\n")
            f.write(f"Défaites: {self.game_results['losses']}\n")
            total_games = self.game_results['wins'] + self.game_results['losses']
            if total_games > 0:
                win_rate = self.game_results['wins'] / total_games * 100
                f.write(f"Taux de victoire: {win_rate:.1f}%\n")
            f.write(f"Temps total: {self.game_results['total_time']:.2f}s\n\n")
            
            # Statistiques de temps
            if self.metrics['execution_times']:
                f.write("2. STATISTIQUES DE TEMPS D'EXÉCUTION\n")
                f.write("-" * 60 + "\n")
                times = self.metrics['execution_times']
                f.write(f"Moyenne: {np.mean(times):.3f}s\n")
                f.write(f"Médiane: {np.median(times):.3f}s\n")
                f.write(f"Écart-type: {np.std(times):.3f}s\n")
                f.write(f"Minimum: {np.min(times):.3f}s\n")
                f.write(f"Maximum: {np.max(times):.3f}s\n")
                f.write(f"95e percentile: {np.percentile(times, 95):.3f}s\n\n")
            
            # Efficacité de recherche
            if self.metrics['nodes_explored']:
                f.write("3. EFFICACITÉ DE LA RECHERCHE\n")
                f.write("-" * 60 + "\n")
                f.write(f"Nœuds moyens explorés: {np.mean(self.metrics['nodes_explored']):.0f}\n")
                f.write(f"Nœuds max explorés: {np.max(self.metrics['nodes_explored']):.0f}\n")
                
                if self.metrics['tt_hits'] and self.metrics['tt_misses']:
                    total_tt = np.sum(self.metrics['tt_hits']) + np.sum(self.metrics['tt_misses'])
                    hit_rate = np.sum(self.metrics['tt_hits']) / total_tt * 100
                    f.write(f"Taux de hit TT: {hit_rate:.1f}%\n")
                
                if self.metrics['alpha_cutoffs']:
                    f.write(f"Coupures alpha-beta moyennes: {np.mean(self.metrics['alpha_cutoffs']):.1f}\n")
                f.write("\n")
            
            # Métriques stratégiques
            if self.metrics['my_distances'] and self.metrics['opp_distances']:
                f.write("4. MÉTRIQUES STRATÉGIQUES\n")
                f.write("-" * 60 + "\n")
                f.write(f"Distance moyenne (nous): {np.mean(self.metrics['my_distances']):.2f}\n")
                f.write(f"Distance moyenne (adversaire): {np.mean(self.metrics['opp_distances']):.2f}\n")
                
                advantage = np.array(self.metrics['opp_distances']) - np.array(self.metrics['my_distances'])
                f.write(f"Avantage moyen: {np.mean(advantage):.2f}\n")
                positive_adv = np.sum(advantage > 0)
                f.write(f"Coups avec avantage: {positive_adv}/{len(advantage)} ({positive_adv/len(advantage)*100:.1f}%)\n\n")
            
            # Détection tactique
            f.write("5. DÉTECTION TACTIQUE\n")
            f.write("-" * 60 + "\n")
            f.write(f"Coups gagnants trouvés: {len([x for x in self.metrics['winning_moves_found'] if x])}\n")
            f.write(f"Coups bloquants trouvés: {len([x for x in self.metrics['blocking_moves_found'] if x])}\n\n")
            
            f.write("=" * 60 + "\n")
        
        print(f"Rapport sauvegardé: {save_path}")
    
    def save_metrics_json(self, save_path='metrics_data.json'):
        """Sauvegarde les métriques en JSON pour analyse ultérieure."""
        data = {
            'metrics': self.metrics,
            'game_results': self.game_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Métriques sauvegardées: {save_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Simulation de données de test
    tracker = PerformanceTracker()
    
    # Simuler une partie
    np.random.seed(42)
    for move_num in range(1, 51):
        board_fill = (move_num / 50) * 100
        
        # Temps d'exécution augmente avec la complexité
        exec_time = 0.1 + (move_num / 50) * 0.5 + np.random.normal(0, 0.1)
        exec_time = max(0.05, exec_time)
        
        tracker.record_move(
            execution_time=exec_time,
            move_number=move_num,
            board_fill=board_fill,
            search_depth=2 + (move_num // 15),
            nodes_explored=int(1000 + move_num * 50 + np.random.normal(0, 200)),
            tt_hits=int(500 + np.random.normal(0, 100)),
            tt_misses=int(300 + np.random.normal(0, 50)),
            alpha_cutoffs=int(50 + np.random.normal(0, 10)),
            my_distance=max(0, 10 - move_num * 0.2 + np.random.normal(0, 1)),
            opp_distance=max(0, 10 - move_num * 0.15 + np.random.normal(0, 1)),
            branching_factor=30 - move_num * 0.5,
            evaluation_score=np.random.normal(0, 1000),
            winning_move_found=move_num > 45 and np.random.random() > 0.8,
            blocking_move_found=np.random.random() > 0.7
        )
    
    tracker.update_game_result(won=True, total_time=sum(tracker.metrics['execution_times']))
    
    # Générer tous les graphiques et rapports
    print("Génération des graphiques et rapports...\n")
    tracker.plot_execution_time_vs_game_progress()
    tracker.plot_search_efficiency()
    tracker.plot_strategic_metrics()
    tracker.generate_summary_report()
    tracker.save_metrics_json()
    
    print("\n✅ Analyse terminée! Vérifiez les fichiers générés.")