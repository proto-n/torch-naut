python run_crps.py --expname crps
python run_wcrps.py --expname wcrps
python run_wcrps_mh.py --expname wcrps_mh
python run_wcrps_mh_ml.py --expname wcrps_mh_ml
python run_wcrps_ens.py --expname wcrps_ens
python run_wcrps_ens_mh.py --expname wcrps_ens_mh
python run_wcrps_ens_mh_ml.py --expname wcrps_ens_mh_ml
python run_mdn.py --expname mdn --n_components 100
python run_mdn_gelu.py --expname mdn_gelu --n_components 100
python run_mdn_bnn.py --expname mdn_bnn --n_components 100
python run_mdn_bnn_gelu.py --expname mdn_bnn_gelu --n_components 100
python run_ensembles.py --expname ensembles --n_networks 5
python run_ensembles_gelu.py --expname ensembles_gelu --n_networks 5
python run_disco.py --expanane disco_1_0 --beta 1 --gamma 0
python run_disco.py --expanane disco_1_025 --beta 1 --gamma 0.25
python run_disco.py --expanane disco_1_05 --beta 1 --gamma 0.5