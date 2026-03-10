cd /mnt/d/Python_Project/NetWork_CS536/assignment3/ns-3-dev-master

./ns3 run "scratch/a3 --tcp=TcpNewReno --serversPerTor=16 --flowSizeMB=64 --stopTime=30 --startGap=0.0001 --segmentSize=1448 --out=a3_newreno.csv --verbose=1"
./ns3 run "scratch/a3 --tcp=TcpCubic --serversPerTor=16 --flowSizeMB=64 --stopTime=30 --startGap=0.0001 --segmentSize=1448 --out=a3_cubic.csv --verbose=1"
./ns3 run "scratch/a3 --tcp=TcpA2Linear --serversPerTor=16 --flowSizeMB=64 --stopTime=30 --startGap=0.0001 --segmentSize=1448 --out=a3_a2linear.csv --verbose=1"

python plot_fct.py