"""Generate the SUMO test network: a straight 3-lane motorway, background
flow, and one emergency vehicle. Writes into out/sumo/ and runs netconvert."""
import os
import subprocess

EDGE_LEN = 3000.0
N_LANES = 3
LANE_W = 3.5
V_MAX = 27.78          # 100 km/h background limit
EV_DEPART = 120.0
FLOW_PER_LANE = 1700   # veh/h/lane -> dense but moving
SIM_END = 420.0


def sumo_bin(name="sumo"):
    home = os.environ.get("SUMO_HOME", "")
    cand = os.path.join(home, "bin", name + ".exe")
    return cand if os.path.exists(cand) else name


def write_scenario(d):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "hw.nod.xml"), "w") as f:
        f.write(f"""<nodes>
  <node id="n0" x="0" y="0"/>
  <node id="n1" x="{EDGE_LEN:.0f}" y="0"/>
</nodes>
""")
    with open(os.path.join(d, "hw.edg.xml"), "w") as f:
        f.write(f"""<edges>
  <edge id="hw" from="n0" to="n1" numLanes="{N_LANES}" speed="{V_MAX}" width="{LANE_W}"/>
</edges>
""")
    net = os.path.join(d, "hw.net.xml")
    subprocess.run([sumo_bin("netconvert"),
                    "-n", os.path.join(d, "hw.nod.xml"),
                    "-e", os.path.join(d, "hw.edg.xml"),
                    "-o", net, "--no-turnarounds"], check=True,
                   capture_output=True)

    with open(os.path.join(d, "hw.rou.xml"), "w") as f:
        f.write(f"""<routes>
  <vType id="car" length="4.5" width="1.8" maxSpeed="41" speedFactor="normc(0.92,0.08,0.6,1.2)"
         maxSpeedLat="1.5" latAlignment="center" minGapLat="0.35" lcSublane="1.0"/>
  <vType id="ev" vClass="emergency" length="6.2" width="2.2" maxSpeed="41"
         speedFactor="1.40" color="1,0,0" maxSpeedLat="2.0" latAlignment="center"
         minGapLat="0.25" jmDriveAfterRedTime="300"
         lcSpeedGain="3.0" lcAssertive="1.6" lcPushy="0.6" lcImpatience="0.9"/>
  <route id="r0" edges="hw"/>
  <flow id="bg" type="car" route="r0" begin="0" end="{SIM_END:.0f}"
        vehsPerHour="{FLOW_PER_LANE * N_LANES}" departLane="random" departSpeed="max"
        departPosLat="random"/>
</routes>
""")

    with open(os.path.join(d, "hw.sumocfg"), "w") as f:
        f.write("""<configuration>
  <input>
    <net-file value="hw.net.xml"/>
    <route-files value="hw.rou.xml"/>
  </input>
  <processing>
    <step-length value="0.1"/>
    <lateral-resolution value="0.4"/>
    <collision.action value="warn"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <no-warnings value="true"/>
  </report>
</configuration>
""")
    return os.path.join(d, "hw.sumocfg")
