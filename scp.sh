from_path="/mnt/zfm_18T/fengxiang/HeNan/Data/OBS"
to_path="./"
rsync -zrL --progress -e 'ssh -p9995' fengxiang@210.26.48.58:${from_path}/* ${to_path} > rsync.log
