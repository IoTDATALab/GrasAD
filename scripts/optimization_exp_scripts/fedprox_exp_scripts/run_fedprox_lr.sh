set -e

cd ../..

echo "Run fedopt on synthetic."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/asyn_dp_synthetic_mapas.yaml \
  fedprox.use True \
  fedprox.mu 0.1
