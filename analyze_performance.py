#!/usr/bin/env python3
"""
Script d'analyse de performance pour l'agent Hex.
À utiliser APRÈS la partie pour générer les graphiques et rapports.

Usage:
    python analyze_performance.py performance_data.json
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_performance_data(filename):
    """Charge les données de performance depuis le JSON."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Fichier {filename} non trouvé!")
        print("Assurez-vous d'avoir appelé MyPlayer.save_performance_data() après la partie.")
        sys.exit(1)

def plot_execution_time_analysis(data, save_path='execution_time_analysis.png'):
    """Génère le graphique d'analyse du temps d'exécution."""
    if not data['execution_times']:
        print("⚠️  Aucune donnée de temps d'exécution")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse du Temps d\'Exécution', fontsize=16, fontweight='bold')
    
    move_numbers = data['move_numbers']
    exec_times = data['execution_times']
    
    # 1. Temps vs Numéro de coup
    ax1 = axes[0, 0]
    ax1.plot(move_numbers, exec_times, marker='o', markersize=4, linewidth=1, alpha=0.7, color='#2E86AB')
    ax1.set_xlabel('Numéro du Coup', fontsize=11)
    ax1.set_ylabel('Temps (secondes)', fontsize=11)
    ax1.set_title('Temps d\'Exécution par Coup', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Ligne de tendance polynomiale
    if len(move_numbers) > 3:
        z = np.polyfit(move_numbers, exec_times, 2)
        p = np.poly1d(z)
        ax1.plot(move_numbers, p(move_numbers), "r--", alpha=0.5, linewidth=2, label='Tendance')
        ax1.legend()
    
    # 2. Temps vs % Plateau Rempli
    ax2 = axes[0, 1]
    if data['board_fills']:
        scatter = ax2.scatter(data['board_fills'], exec_times, 
                             alpha=0.6, c=move_numbers, cmap='viridis', s=50)
        ax2.set_xlabel('Plateau Rempli (%)', fontsize=11)
        ax2.set_ylabel('Temps (secondes)', fontsize=11)
        ax2.set_title('Temps vs Complexité du Plateau', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Numéro du Coup', fontsize=10)
    
    # 3. Distribution des temps
    ax3 = axes[1, 0]
    ax3.hist(exec_times, bins=20, alpha=0.7, edgecolor='black', color='#A23B72')
    mean_time = np.mean(exec_times)
    median_time = np.median(exec_times)
    ax3.axvline(mean_time, color='r', linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_time:.3f}s')
    ax3.axvline(median_time, color='g', linestyle='--', linewidth=2, 
               label=f'Médiane: {median_time:.3f}s')
    ax3.set_xlabel('Temps (secondes)', fontsize=11)
    ax3.set_ylabel('Fréquence', fontsize=11)
    ax3.set_title('Distribution des Temps d\'Exécution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Temps cumulé
    ax4 = axes[1, 1]
    cumulative_time = np.cumsum(exec_times)
    ax4.plot(move_numbers, cumulative_time, linewidth=2, color='#F18F01')
    ax4.set_xlabel('Numéro du Coup', fontsize=11)
    ax4.set_ylabel('Temps Cumulé (secondes)', fontsize=11)
    ax4.set_title('Temps Cumulé au Cours de la Partie', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(move_numbers, cumulative_time, alpha=0.3, color='#F18F01')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}")
    plt.close()

def plot_search_efficiency(data, save_path='search_efficiency.png'):
    """Génère le graphique d'efficacité de recherche."""
    if not data['nodes_explored']:
        print("⚠️  Aucune donnée de recherche")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Efficacité de la Recherche Minimax', fontsize=16, fontweight='bold')
    
    move_numbers = data['move_numbers']
    
    # 1. Nœuds explorés vs temps
    ax1 = axes[0, 0]
    ax1.scatter(data['nodes_explored'], data['execution_times'], alpha=0.6, color='#6A4C93', s=50)
    ax1.set_xlabel('Nœuds Explorés', fontsize=11)
    ax1.set_ylabel('Temps (secondes)', fontsize=11)
    ax1.set_title('Temps vs Nœuds Explorés', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Ajouter une ligne de régression
    if len(data['nodes_explored']) > 1:
        z = np.polyfit(data['nodes_explored'], data['execution_times'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(data['nodes_explored']), max(data['nodes_explored']), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Régression linéaire')
        ax1.legend()
    
    # 2. Taux de hit du TT
    ax2 = axes[0, 1]
    if data['tt_hits'] and data['tt_misses']:
        tt_total = np.array(data['tt_hits']) + np.array(data['tt_misses'])
        tt_rate = np.where(tt_total > 0, np.array(data['tt_hits']) / tt_total * 100, 0)
        ax2.plot(move_numbers, tt_rate, marker='o', markersize=4, color='#1B9AAA', linewidth=2)
        ax2.set_xlabel('Numéro du Coup', fontsize=11)
        ax2.set_ylabel('Taux de Hit TT (%)', fontsize=11)
        ax2.set_title('Efficacité du Transposition Table', fontsize=12, fontweight='bold')
        mean_rate = np.mean(tt_rate[tt_rate > 0]) if np.any(tt_rate > 0) else 0
        ax2.axhline(mean_rate, color='r', linestyle='--', linewidth=2,
                   label=f'Moyenne: {mean_rate:.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    
    # 3. Profondeur de recherche
    ax3 = axes[1, 0]
    if data['search_depths']:
        depths = data['search_depths']
        ax3.plot(move_numbers, depths, marker='s', markersize=5, linewidth=2, color='#DD6E42')
        ax3.set_xlabel('Numéro du Coup', fontsize=11)
        ax3.set_ylabel('Profondeur', fontsize=11)
        ax3.set_title('Profondeur de Recherche Adaptative', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yticks(range(int(min(depths)), int(max(depths))+1))
        
        # Ajouter les zones de couleur pour les phases
        ax3.axhspan(0, 2, alpha=0.1, color='green', label='Ouverture')
        ax3.axhspan(2, 3, alpha=0.1, color='yellow', label='Milieu')
        ax3.axhspan(3, 6, alpha=0.1, color='red', label='Fin')
        ax3.legend(loc='upper left', fontsize=9)
    
    # 4. Coupures Alpha-Beta
    ax4 = axes[1, 1]
    if data['alpha_cutoffs']:
        ax4.bar(move_numbers, data['alpha_cutoffs'], alpha=0.7, color='#E63946')
        ax4.set_xlabel('Numéro du Coup', fontsize=11)
        ax4.set_ylabel('Nombre de Coupures', fontsize=11)
        ax4.set_title('Efficacité de l\'Élagage Alpha-Beta', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Ligne de moyenne
        mean_cutoffs = np.mean(data['alpha_cutoffs'])
        ax4.axhline(mean_cutoffs, color='blue', linestyle='--', linewidth=2,
                   label=f'Moyenne: {mean_cutoffs:.1f}')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}")
    plt.close()

def plot_strategic_metrics(data, save_path='strategic_metrics.png'):
    """Génère le graphique des métriques stratégiques."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse Stratégique', fontsize=16, fontweight='bold')
    
    move_numbers = data['move_numbers']
    
    # 1. Distances shortest path
    ax1 = axes[0, 0]
    if data['my_distances'] and data['opp_distances']:
        ax1.plot(move_numbers, data['my_distances'], 
                label='Notre Distance', marker='o', linewidth=2, color='#2A9D8F')
        ax1.plot(move_numbers, data['opp_distances'], 
                label='Distance Adversaire', marker='s', linewidth=2, color='#E76F51')
        ax1.set_xlabel('Numéro du Coup', fontsize=11)
        ax1.set_ylabel('Distance', fontsize=11)
        ax1.set_title('Évolution des Distances Shortest Path', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(move_numbers, data['my_distances'], alpha=0.2, color='#2A9D8F')
        ax1.fill_between(move_numbers, data['opp_distances'], alpha=0.2, color='#E76F51')
    
    # 2. Avantage stratégique
    ax2 = axes[0, 1]
    if data['my_distances'] and data['opp_distances']:
        advantage = np.array(data['opp_distances']) - np.array(data['my_distances'])
        colors = ['#2D6A4F' if a > 0 else '#9D0208' for a in advantage]
        ax2.bar(move_numbers, advantage, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=2)
        ax2.set_xlabel('Numéro du Coup', fontsize=11)
        ax2.set_ylabel('Avantage (Dist. Opp - Notre Dist.)', fontsize=11)
        ax2.set_title('Avantage Stratégique par Coup', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Statistiques
        positive_count = np.sum(advantage > 0)
        total_count = len(advantage)
        ax2.text(0.02, 0.98, f'Avantage: {positive_count}/{total_count} coups ({positive_count/total_count*100:.1f}%)',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 3. Scores d'évaluation
    ax3 = axes[1, 0]
    if data['evaluation_scores']:
        scores = data['evaluation_scores']
        ax3.plot(move_numbers, scores, linewidth=2, color='#7209B7')
        ax3.set_xlabel('Numéro du Coup', fontsize=11)
        ax3.set_ylabel('Score d\'Évaluation', fontsize=11)
        ax3.set_title('Évolution du Score d\'Évaluation', fontsize=12, fontweight='bold')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(move_numbers, scores, 0, where=(np.array(scores) > 0), 
                        alpha=0.3, color='green', interpolate=True)
        ax3.fill_between(move_numbers, scores, 0, where=(np.array(scores) <= 0), 
                        alpha=0.3, color='red', interpolate=True)
    
    # 4. Facteur de branchement
    ax4 = axes[1, 1]
    if data['branching_factors']:
        bf = data['branching_factors']
        ax4.plot(move_numbers, bf, marker='o', linewidth=2, color='#F77F00')
        ax4.set_xlabel('Numéro du Coup', fontsize=11)
        ax4.set_ylabel('Facteur de Branchement', fontsize=11)
        ax4.set_title('Complexité de l\'Arbre de Recherche', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(move_numbers, bf, alpha=0.3, color='#F77F00')
        
        # Moyenne
        mean_bf = np.mean(bf)
        ax4.axhline(mean_bf, color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {mean_bf:.1f}')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}")
    plt.close()

def generate_text_report(data, save_path='performance_report.txt'):
    """Génère un rapport textuel détaillé."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(" " * 20 + "RAPPORT DE PERFORMANCE - AGENT HEX\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Résultats généraux
        f.write("1. INFORMATIONS GÉNÉRALES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nombre de coups joués: {len(data['execution_times'])}\n")
        if 'total_game_time' in data:
            f.write(f"Durée totale de la partie: {data['total_game_time']:.2f}s\n")
        f.write(f"Coups gagnants détectés: {data.get('winning_moves', 0)}\n")
        f.write(f"Coups bloquants détectés: {data.get('blocking_moves', 0)}\n\n")
        
        # 2. Statistiques de temps
        if data['execution_times']:
            f.write("2. STATISTIQUES DE TEMPS D'EXÉCUTION\n")
            f.write("-" * 70 + "\n")
            times = data['execution_times']
            f.write(f"Moyenne:        {np.mean(times):.4f}s\n")
            f.write(f"Médiane:        {np.median(times):.4f}s\n")
            f.write(f"Écart-type:     {np.std(times):.4f}s\n")
            f.write(f"Minimum:        {np.min(times):.4f}s\n")
            f.write(f"Maximum:        {np.max(times):.4f}s\n")
            f.write(f"25e percentile: {np.percentile(times, 25):.4f}s\n")
            f.write(f"75e percentile: {np.percentile(times, 75):.4f}s\n")
            f.write(f"95e percentile: {np.percentile(times, 95):.4f}s\n\n")
        
        # 3. Efficacité de recherche
        if data['nodes_explored']:
            f.write("3. EFFICACITÉ DE LA RECHERCHE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Nœuds explorés (moyenne):   {np.mean(data['nodes_explored']):.0f}\n")
            f.write(f"Nœuds explorés (médiane):   {np.median(data['nodes_explored']):.0f}\n")
            f.write(f"Nœuds explorés (max):       {np.max(data['nodes_explored']):.0f}\n")
            f.write(f"Nœuds explorés (total):     {np.sum(data['nodes_explored']):.0f}\n\n")
            
            if data['tt_hits'] and data['tt_misses']:
                total_tt = np.sum(data['tt_hits']) + np.sum(data['tt_misses'])
                hit_rate = np.sum(data['tt_hits']) / total_tt * 100 if total_tt > 0 else 0
                f.write(f"Hits Transposition Table:   {np.sum(data['tt_hits']):.0f}\n")
                f.write(f"Misses Transposition Table: {np.sum(data['tt_misses']):.0f}\n")
                f.write(f"Taux de hit TT:             {hit_rate:.2f}%\n\n")
            
            if data['alpha_cutoffs']:
                f.write(f"Coupures alpha-beta (moyenne): {np.mean(data['alpha_cutoffs']):.1f}\n")
                f.write(f"Coupures alpha-beta (total):   {np.sum(data['alpha_cutoffs']):.0f}\n\n")
            
            if data['search_depths']:
                f.write(f"Profondeur de recherche (moyenne): {np.mean(data['search_depths']):.2f}\n")
                f.write(f"Profondeur de recherche (max):     {np.max(data['search_depths'])}\n\n")
        
        # 4. Métriques stratégiques
        if data['my_distances'] and data['opp_distances']:
            f.write("4. MÉTRIQUES STRATÉGIQUES\n")
            f.write("-" * 70 + "\n")
            my_dists = data['my_distances']
            opp_dists = data['opp_distances']
            
            f.write(f"Distance moyenne (nous):        {np.mean(my_dists):.3f}\n")
            f.write(f"Distance moyenne (adversaire):  {np.mean(opp_dists):.3f}\n")
            f.write(f"Distance finale (nous):         {my_dists[-1]:.3f}\n")
            f.write(f"Distance finale (adversaire):   {opp_dists[-1]:.3f}\n\n")
            
            advantage = np.array(opp_dists) - np.array(my_dists)
            f.write(f"Avantage moyen:                 {np.mean(advantage):.3f}\n")
            f.write(f"Avantage médian:                {np.median(advantage):.3f}\n")
            
            positive_adv = np.sum(advantage > 0)
            total_moves = len(advantage)
            f.write(f"Coups avec avantage:            {positive_adv}/{total_moves} ")
            f.write(f"({positive_adv/total_moves*100:.1f}%)\n\n")
        
        # 5. Performance par phase de jeu
        if data['board_fills'] and data['execution_times']:
            f.write("5. PERFORMANCE PAR PHASE DE JEU\n")
            f.write("-" * 70 + "\n")
            
            fills = np.array(data['board_fills'])
            times = np.array(data['execution_times'])
            
            # Ouverture (0-33%)
            opening_mask = fills < 33
            if np.any(opening_mask):
                f.write(f"Ouverture (0-33% rempli):\n")
                f.write(f"  Temps moyen: {np.mean(times[opening_mask]):.4f}s\n")
                f.write(f"  Coups joués: {np.sum(opening_mask)}\n\n")
            
            # Milieu (33-66%)
            middle_mask = (fills >= 33) & (fills < 66)
            if np.any(middle_mask):
                f.write(f"Milieu de partie (33-66% rempli):\n")
                f.write(f"  Temps moyen: {np.mean(times[middle_mask]):.4f}s\n")
                f.write(f"  Coups joués: {np.sum(middle_mask)}\n\n")
            
            # Fin (66-100%)
            endgame_mask = fills >= 66
            if np.any(endgame_mask):
                f.write(f"Fin de partie (66-100% rempli):\n")
                f.write(f"  Temps moyen: {np.mean(times[endgame_mask]):.4f}s\n")
                f.write(f"  Coups joués: {np.sum(endgame_mask)}\n\n")
        
        # 6. Analyse des pics de performance
        if data['execution_times']:
            f.write("6. ANALYSE DES PICS DE PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            times = np.array(data['execution_times'])
            move_nums = data['move_numbers']
            
            # Top 5 coups les plus lents
            slowest_indices = np.argsort(times)[-5:][::-1]
            f.write("5 coups les plus lents:\n")
            for idx in slowest_indices:
                f.write(f"  Coup #{move_nums[idx]}: {times[idx]:.4f}s\n")
            
            f.write("\n")
            
            # Top 5 coups les plus rapides
            fastest_indices = np.argsort(times)[:5]
            f.write("5 coups les plus rapides:\n")
            for idx in fastest_indices:
                f.write(f"  Coup #{move_nums[idx]}: {times[idx]:.4f}s\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Fin du rapport\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Rapport sauvegardé: {save_path}")

def main():
    """Fonction principale."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <fichier_json>")
        print("Exemple: python analyze_performance.py performance_data.json")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print("\n" + "="*70)
    print(" " * 15 + "ANALYSE DE PERFORMANCE - AGENT HEX")
    print("="*70 + "\n")
    
    # Charger les données
    print(f"📂 Chargement des données depuis {filename}...")
    data = load_performance_data(filename)
    
    print(f"✅ {len(data['execution_times'])} coups chargés\n")
    
    # Générer les graphiques
    print("📊 Génération des graphiques...\n")
    
    plot_execution_time_analysis(data, 'hex_execution_time_analysis.png')
    plot_search_efficiency(data, 'hex_search_efficiency.png')
    plot_strategic_metrics(data, 'hex_strategic_metrics.png')
    
    # Générer le rapport textuel
    print("\n📝 Génération du rapport textuel...\n")
    generate_text_report(data, 'hex_performance_report.txt')
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE!")
    print("="*70)
    print("\nFichiers générés:")
    print("  📈 hex_execution_time_analysis.png")
    print("  📈 hex_search_efficiency.png")
    print("  📈 hex_strategic_metrics.png")
    print("  📄 hex_performance_report.txt")
    print("\n")

if __name__ == "__main__":
    main()