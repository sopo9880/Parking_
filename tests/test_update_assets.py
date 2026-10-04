import unittest
from unittest.mock import patch
import update_manager

class ReleaseAssetTests(unittest.TestCase):
    def test_evidence_zip_never_becomes_program_update(self):
        data={'tag_name':'v16.5.2','assets':[
            {'name':'ParkingResearchAgent-v16.5.2.zip'},
            {'name':'ParkingResearchAgent-v16.5.2.zip.sha256'},
            {'name':'ParkingResearchAgent-v16.5.2-CNRPark-External-Baseline.zip'},
            {'name':'ParkingResearchAgent-v16.5.2-CNRPark-External-Baseline.zip.sha256'}]}
        with patch.object(update_manager,'_get_json',return_value=data):
            result=update_manager.check_latest({})
        self.assertEqual(result['zip_asset']['name'],'ParkingResearchAgent-v16.5.2.zip')
        self.assertEqual(result['sha_asset']['name'],'ParkingResearchAgent-v16.5.2.zip.sha256')

if __name__=='__main__': unittest.main()
