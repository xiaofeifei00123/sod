from_path="/home/fengxiang/FX/TPV_simulation_single/Figure_TBB/TBB_obs_distribution/"
to_path="./"
rsync -zrL --progress -e 'ssh -p9995' fengxiang@210.26.48.58:${from_path}/* ${to_path} > rsync.log
