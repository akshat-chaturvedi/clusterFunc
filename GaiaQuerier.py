import os
from astroquery.gaia import Gaia
import dotenv

class GaiaQuerier:
    def __init__(self, clusterName):
        self.clusterName = clusterName

        dotenv.load_dotenv('.env')
        print("---->Gaia credentials found")

        Gaia.login(user=os.getenv("GAIA_USERID"), password=os.getenv("GAIA_PASSWORD"))

        job = Gaia.upload_table(upload_resource=f"gData/{self.clusterName}.csv",
                                table_name=f"{self.clusterName.lower()}",
                                format="csv")

        job1 = Gaia.launch_job("SELECT TOP 100000 "
                               f"source_id, hip.{self.clusterName.lower()}_oid, parallax, parallax_error, gaia.ra, "
                               f"gaia.dec,"
                               "phot_g_mean_mag, pmra, "
                               "pmra_error, pmdec, pmdec_error, DISTANCE( "
                               "POINT(hip.ra, hip.dec), "
                               "POINT(gaia.ra, gaia.dec) "
                               ") AS ang_sep "
                               f"FROM user_achatu01.{self.clusterName.lower()} AS hip "
                               "JOIN gaiadr3.gaia_source AS gaia "
                               "ON 1 = CONTAINS( "
                               "POINT(hip.ra, hip.dec), "
                               "CIRCLE(gaia.ra, gaia.dec, 0.00028) "
                               ") "
                               "WHERE ABS(gaia.phot_g_mean_mag - hip.g) < 0.9",
                               dump_to_file=False)

        job1_results = job1.get_results()
        job1_results.write(f"GAIAData/{self.clusterName}.csv", format="csv")

        Gaia.logout()
        print("---->Gaia Querying Successful")
