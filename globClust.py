from GCAnalyzer import GCAnalyzer
import logging

def main():
    a = GCAnalyzer()
    a.dataCleaner()
    a.dereddening()
    a.uniqueMatches()
    a.clusterMembershipChecks()
    a.HBParams()
    a.CMDPlotter()
    a.dataSaver()


if __name__ == "__main__":
    logging.basicConfig(filename='clusterLogs.log',
                        encoding='utf-8',
                        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO)
    main()
