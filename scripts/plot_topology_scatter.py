
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def main():
    print("Generating 3D Topology Scatter Plot...")
    
    try:
        df = pd.read_csv("disagreement_topology.csv")
    except FileNotFoundError:
        print("Error: disagreement_topology.csv not found. Run scripts/disagreement_analysis.py first.")
        return

    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color mapping for categories
    categories = df['category'].unique()
    colors = sns.color_palette("hls", len(categories))
    color_map = dict(zip(categories, colors))

    for cat in categories:
        sub = df[df['category'] == cat]
        ax.scatter(
            sub['dist_lit_intent'], 
            sub['dist_lit_facet'], 
            sub['dist_facet_intent'],
            label=cat,
            alpha=0.6,
            edgecolors='w',
            s=60
        )

    # Label axes
    ax.set_xlabel('L-I Axis (Literal vs Intent)')
    ax.set_ylabel('L-F Axis (Literal vs Facet)')
    ax.set_zlabel('F-I Axis (Facet vs Intent)')
    
    plt.title("Disagreement Topology: Query Clusters in 3D Disagreement Space\n(Points = Individual Queries, Colored by CLAMBER Label)")
    plt.legend()
    
    # Save a few angles to help with 3D visualization
    output_path = "disagreement_topology_3d.png"
    plt.savefig(output_path, dpi=300)
    
    # Also save a 2D projection (L-I vs L-F) for easier viewing
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x='dist_lit_intent', 
        y='dist_lit_facet', 
        hue='category', 
        style='category',
        alpha=0.7,
        s=100
    )
    plt.title("2D Projection of Disagreement Topology (L-I vs L-F)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("disagreement_topology_2d.png", dpi=300)

    print(f"Topology graphics saved to {output_path} and disagreement_topology_2d.png")

if __name__ == "__main__":
    main()
