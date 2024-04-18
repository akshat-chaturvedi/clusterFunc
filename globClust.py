from GCAnalyzer import GCAnalyzer
import logging
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def get_file_creation_date(cluster_name):
    file_name = f"candStars/candStarsWithProbs/{cluster_name}_candStarsWithProb_April.dat"
    if os.path.isfile(file_name):
        creation_time = os.path.getctime(file_name)
        return datetime.datetime.fromtimestamp(creation_time).strftime('%m/%d/%Y')
    else:
        return None

def inventoryChecker():
    file_path = "goodClusters.txt"

    with open(file_path, "r") as file:
        # Read the names
        cluster_names = file.readlines()

    # Modify the names based on the condition
    modified_names = []
    for name in cluster_names:
        name = name.strip()  # Remove leading/trailing whitespaces
        creation_date = get_file_creation_date(name)
        if creation_date:
            name += f"-->DONE-{creation_date}"
        modified_names.append(name)

    # Write back to the file
    with open("analysisInventory.txt", "w") as file:
        file.write("\n".join(modified_names))

    print("Cluster names have been checked and modified.")


def candStarPlotter():
    data_dir = 'candStars/candStarsWithProbs'

    # Columns you want to extract
    column1_name = 3
    column2_name = 4
    column3_name = 6

    # Lists to store column values
    column1_values = []
    column2_values = []
    column3_values = []

    # Loop through each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.dat'):
            file_path = os.path.join(data_dir, filename)

            # Read the CSV file
            df = pd.read_csv(file_path, sep="\t", skiprows=5, header=None)

            column1_values.extend(df[column1_name].tolist())
            column2_values.extend(df[column2_name].tolist())
            column3_values.extend(df[column3_name].tolist())

    # V vs B-V Plot
    fig, ax = plt.subplots()
    ax.scatter(column1_values, column3_values, c='k', s=1)
    ax.set_xlim(-1.1, 0.1)
    ax.set_ylim(7, -5)
    ax.set_xlabel('($B-V$)$_0$', fontsize=14)
    ax.set_ylabel('M$_V$', style="italic", fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    ax.set_title(f"Candidate Stars", fontsize=16)
    fig.savefig(f'Plots/DBSCAN/candStarPlots/candStars_VBV.pdf', bbox_inches='tight', dpi=300)

    # u vs B-V Plot
    fig, ax = plt.subplots()
    ax.scatter(column1_values, column2_values, c='k', s=1)
    ax.set_xlim(-1.1, 0.1)
    ax.set_ylim(7, -5)
    ax.set_xlabel('($B-V$)$_0$', fontsize=14)
    ax.set_ylabel('M$_u$', style="italic", fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    ax.set_title(f"Candidate Stars", fontsize=16)
    fig.savefig(f'Plots/DBSCAN/candStarPlots/candStars_UBV.pdf', bbox_inches='tight', dpi=300)
    print(f"Candidate Star CMDs plotted")


def main():
    a = GCAnalyzer()
    a.dataCleaner()
    a.dereddening()
    a.uniqueMatches()
    a.clusterMembershipChecks()
    a.HBParams()
    a.CMDPlotter()
    a.dataSaver()

    inventoryChecker()
    candStarPlotter()


if __name__ == "__main__":
    logging.basicConfig(filename='clusterLogs.log',
                        encoding='utf-8',
                        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO)
    main()
