from_path="/mnt/zfm_18T/fengxiang/HeNan/Draw/picture_rain/rain_6d/"
to_path="./"
rsync -zrL --progress -e 'ssh -p9995' fengxiang@210.26.48.58:${from_path}/11colors.txt ${to_path} > rsync.log
