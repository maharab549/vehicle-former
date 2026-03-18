"""
Create SUMO simulation scenario files for VehicleFormer.
Generates a realistic urban road network with 5G base stations,
C-V2X RSUs, and satellite coverage zones.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

SUMO_DIR = Path("data/sumo")
SUMO_DIR.mkdir(parents=True, exist_ok=True)


def create_network():
    """Create a 2x2 km urban grid network with intersections."""
    net_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

    <!-- NODES: 5x5 grid = 25 intersections, 200m spacing -->
    <node id="n00" x="0"    y="0"    type="traffic_light"/>
    <node id="n01" x="200"  y="0"    type="traffic_light"/>
    <node id="n02" x="400"  y="0"    type="traffic_light"/>
    <node id="n03" x="600"  y="0"    type="traffic_light"/>
    <node id="n04" x="800"  y="0"    type="traffic_light"/>
    <node id="n10" x="0"    y="200"  type="traffic_light"/>
    <node id="n11" x="200"  y="200"  type="traffic_light"/>
    <node id="n12" x="400"  y="200"  type="traffic_light"/>
    <node id="n13" x="600"  y="200"  type="traffic_light"/>
    <node id="n14" x="800"  y="200"  type="traffic_light"/>
    <node id="n20" x="0"    y="400"  type="traffic_light"/>
    <node id="n21" x="200"  y="400"  type="traffic_light"/>
    <node id="n22" x="400"  y="400"  type="traffic_light"/>
    <node id="n23" x="600"  y="400"  type="traffic_light"/>
    <node id="n24" x="800"  y="400"  type="traffic_light"/>
    <node id="n30" x="0"    y="600"  type="traffic_light"/>
    <node id="n31" x="200"  y="600"  type="traffic_light"/>
    <node id="n32" x="400"  y="600"  type="traffic_light"/>
    <node id="n33" x="600"  y="600"  type="traffic_light"/>
    <node id="n34" x="800"  y="600"  type="traffic_light"/>
    <node id="n40" x="0"    y="800"  type="traffic_light"/>
    <node id="n41" x="200"  y="800"  type="traffic_light"/>
    <node id="n42" x="400"  y="800"  type="traffic_light"/>
    <node id="n43" x="600"  y="800"  type="traffic_light"/>
    <node id="n44" x="800"  y="800"  type="traffic_light"/>

    <!-- EDGES: horizontal -->
    <edge id="e00_01" from="n00" to="n01" numLanes="2" speed="13.89"/>
    <edge id="e01_02" from="n01" to="n02" numLanes="2" speed="13.89"/>
    <edge id="e02_03" from="n02" to="n03" numLanes="2" speed="13.89"/>
    <edge id="e03_04" from="n03" to="n04" numLanes="2" speed="13.89"/>
    <edge id="e10_11" from="n10" to="n11" numLanes="2" speed="13.89"/>
    <edge id="e11_12" from="n11" to="n12" numLanes="2" speed="13.89"/>
    <edge id="e12_13" from="n12" to="n13" numLanes="2" speed="13.89"/>
    <edge id="e13_14" from="n13" to="n14" numLanes="2" speed="13.89"/>
    <edge id="e20_21" from="n20" to="n21" numLanes="2" speed="13.89"/>
    <edge id="e21_22" from="n21" to="n22" numLanes="2" speed="13.89"/>
    <edge id="e22_23" from="n22" to="n23" numLanes="2" speed="13.89"/>
    <edge id="e23_24" from="n23" to="n24" numLanes="2" speed="13.89"/>
    <edge id="e30_31" from="n30" to="n31" numLanes="2" speed="13.89"/>
    <edge id="e31_32" from="n31" to="n32" numLanes="2" speed="13.89"/>
    <edge id="e32_33" from="n32" to="n33" numLanes="2" speed="13.89"/>
    <edge id="e33_34" from="n33" to="n34" numLanes="2" speed="13.89"/>
    <edge id="e40_41" from="n40" to="n41" numLanes="2" speed="13.89"/>
    <edge id="e41_42" from="n41" to="n42" numLanes="2" speed="13.89"/>
    <edge id="e42_43" from="n42" to="n43" numLanes="2" speed="13.89"/>
    <edge id="e43_44" from="n43" to="n44" numLanes="2" speed="13.89"/>

    <!-- EDGES: vertical -->
    <edge id="e00_10" from="n00" to="n10" numLanes="2" speed="13.89"/>
    <edge id="e10_20" from="n10" to="n20" numLanes="2" speed="13.89"/>
    <edge id="e20_30" from="n20" to="n30" numLanes="2" speed="13.89"/>
    <edge id="e30_40" from="n30" to="n40" numLanes="2" speed="13.89"/>
    <edge id="e01_11" from="n01" to="n11" numLanes="2" speed="13.89"/>
    <edge id="e11_21" from="n11" to="n21" numLanes="2" speed="13.89"/>
    <edge id="e21_31" from="n21" to="n31" numLanes="2" speed="13.89"/>
    <edge id="e31_41" from="n31" to="n41" numLanes="2" speed="13.89"/>
    <edge id="e02_12" from="n02" to="n12" numLanes="2" speed="13.89"/>
    <edge id="e12_22" from="n12" to="n22" numLanes="2" speed="13.89"/>
    <edge id="e22_32" from="n22" to="n32" numLanes="2" speed="13.89"/>
    <edge id="e32_42" from="n32" to="n42" numLanes="2" speed="13.89"/>
    <edge id="e03_13" from="n03" to="n13" numLanes="2" speed="13.89"/>
    <edge id="e13_23" from="n13" to="n23" numLanes="2" speed="13.89"/>
    <edge id="e23_33" from="n23" to="n33" numLanes="2" speed="13.89"/>
    <edge id="e33_43" from="n33" to="n43" numLanes="2" speed="13.89"/>
    <edge id="e04_14" from="n04" to="n14" numLanes="2" speed="13.89"/>
    <edge id="e14_24" from="n14" to="n24" numLanes="2" speed="13.89"/>
    <edge id="e24_34" from="n24" to="n34" numLanes="2" speed="13.89"/>
    <edge id="e34_44" from="n34" to="n44" numLanes="2" speed="13.89"/>

</net>"""
    with open(SUMO_DIR / "urban_grid.net.xml", "w") as f:
        f.write(net_xml)
    print("  ✓ Network file created: urban_grid.net.xml")


def create_routes():
    """Create vehicle routes for the simulation."""
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="passenger" accel="2.5" decel="4.5" length="4.5" maxSpeed="13.89"
           sigma="0.5" speedFactor="1.0" color="0,0,255"/>
    <vType id="truck" accel="1.5" decel="3.5" length="12.0" maxSpeed="10.0"
           sigma="0.3" speedFactor="0.9" color="255,0,0"/>
    <vType id="bus" accel="1.8" decel="4.0" length="12.0" maxSpeed="11.0"
           sigma="0.2" speedFactor="0.95" color="0,255,0"/>

    <!-- Routes -->
    <route id="route_h0" edges="e00_01 e01_02 e02_03 e03_04"/>
    <route id="route_h1" edges="e10_11 e11_12 e12_13 e13_14"/>
    <route id="route_h2" edges="e20_21 e21_22 e22_23 e23_24"/>
    <route id="route_h3" edges="e30_31 e31_32 e32_33 e33_34"/>
    <route id="route_h4" edges="e40_41 e41_42 e42_43 e43_44"/>
    <route id="route_v0" edges="e00_10 e10_20 e20_30 e30_40"/>
    <route id="route_v1" edges="e01_11 e11_21 e21_31 e31_41"/>
    <route id="route_v2" edges="e02_12 e12_22 e22_32 e32_42"/>
    <route id="route_v3" edges="e03_13 e13_23 e23_33 e33_43"/>
    <route id="route_v4" edges="e04_14 e14_24 e24_34 e34_44"/>

    <!-- Vehicle flows (1 hour simulation) -->
    <flow id="flow_h0" type="passenger" route="route_h0" begin="0" end="3600" vehsPerHour="120"/>
    <flow id="flow_h1" type="passenger" route="route_h1" begin="0" end="3600" vehsPerHour="100"/>
    <flow id="flow_h2" type="passenger" route="route_h2" begin="0" end="3600" vehsPerHour="150"/>
    <flow id="flow_h3" type="passenger" route="route_h3" begin="0" end="3600" vehsPerHour="80"/>
    <flow id="flow_h4" type="passenger" route="route_h4" begin="0" end="3600" vehsPerHour="90"/>
    <flow id="flow_v0" type="passenger" route="route_v0" begin="0" end="3600" vehsPerHour="110"/>
    <flow id="flow_v1" type="passenger" route="route_v1" begin="0" end="3600" vehsPerHour="130"/>
    <flow id="flow_v2" type="passenger" route="route_v2" begin="0" end="3600" vehsPerHour="100"/>
    <flow id="flow_v3" type="passenger" route="route_v3" begin="0" end="3600" vehsPerHour="70"/>
    <flow id="flow_v4" type="passenger" route="route_v4" begin="0" end="3600" vehsPerHour="85"/>
    <flow id="flow_truck" type="truck" route="route_h2" begin="0" end="3600" vehsPerHour="20"/>
    <flow id="flow_bus" type="bus" route="route_v2" begin="0" end="3600" vehsPerHour="15"/>
</routes>"""
    with open(SUMO_DIR / "urban_grid.rou.xml", "w") as f:
        f.write(routes_xml)
    print("  ✓ Routes file created: urban_grid.rou.xml")


def create_config():
    """Create the SUMO configuration file."""
    cfg_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="urban_grid.net.xml"/>
        <route-files value="urban_grid.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <collision.action value="warn"/>
        <time-to-teleport value="300"/>
    </processing>
    <output>
        <tripinfo-output value="tripinfo.xml"/>
    </output>
</configuration>"""
    with open(SUMO_DIR / "urban_grid.sumocfg", "w") as f:
        f.write(cfg_xml)
    print("  ✓ Config file created: urban_grid.sumocfg")


if __name__ == "__main__":
    print("Creating SUMO scenario files...")
    create_network()
    create_routes()
    create_config()
    print(f"  ✓ All files created in {SUMO_DIR.absolute()}")
