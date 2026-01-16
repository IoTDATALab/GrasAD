set -e

cd ../..

echo "Run fedprox on shakespeare."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/asyn_dp_shakespeare_mapas.yaml \
  fedprox.use True \
  fedprox.mu 0.1
