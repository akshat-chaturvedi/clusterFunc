from GCAnalyzer import GCAnalyzer
import logging
import os
import datetime

def get_file_creation_date(cluster_name):
    file_name = f"candStars/candStarsWithProbs/{cluster_name}_candStarsWithProb_April.dat"
    if os.path.isfile(file_name):
        creation_time = os.path.getctime(file_name)
        return datetime.datetime.fromtimestamp(creation_time).strftime('%m/%d/%Y')
    else:
        return None


def main():
    a = GCAnalyzer()
    a.dataCleaner()
    a.dereddening()
    a.uniqueMatches()
    a.clusterMembershipChecks()
    a.HBParams()
    a.CMDPlotter()
    a.dataSaver()

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


if __name__ == "__main__":
    logging.basicConfig(filename='clusterLogs.log',
                        encoding='utf-8',
                        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO)
    main()
