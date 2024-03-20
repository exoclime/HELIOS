# ==============================================================================
# Module with a data base of species used in HELIOS
# Copyright (C) 2020 - 2022 Matej Malik
# ==============================================================================
# This file is part of HELIOS.
#
#     HELIOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     HELIOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You find a copy of the GNU General Public License in the main
#     HELIOS directory under <license.txt>. If not, see
#     <http://www.gnu.org/licenses/>.
# ==============================================================================


class Species_db(object):

    def __init__(self, name, fc_name, weight):

        self.name = name
        self.fc_name = fc_name  # designation in Fastchem
        self.weight = weight  # weight in AMU or g/mol


species_lib = {}

# neutral molecules
species_lib["CO2"] = Species_db(name="CO2", fc_name="C1O2", weight=44.01)
species_lib["H2O"] = Species_db(name="H2O", fc_name="H2O1", weight=18.0153)
species_lib["CO"] = Species_db(name="CO", fc_name="C1O1", weight=28.01)
species_lib["O2"] = Species_db(name="O2", fc_name="O2", weight=31.9988)
species_lib["CH4"] = Species_db(name="CH4", fc_name="C1H4", weight=16.04)
species_lib["HCN"] = Species_db(name="HCN", fc_name="C1H1N1", weight=27.0253)
species_lib["NH3"] = Species_db(name="NH3", fc_name="H3N1", weight=17.031)
species_lib["H2S"] = Species_db(name="H2S", fc_name="H2S1", weight=34.081)
species_lib["PH3"] = Species_db(name="PH3", fc_name="H3P1", weight=33.99758)
species_lib["O3"] = Species_db(name="O3", fc_name="O3", weight=47.9982)
species_lib["O3_IR"] = Species_db(name="O3_IR", fc_name="O3", weight=47.9982)
species_lib["O3_UV"] = Species_db(name="O3_UV", fc_name="O3", weight=47.9982)
species_lib["NO"] = Species_db(name="NO", fc_name="N1O1", weight=30.01)
species_lib["SO2"] = Species_db(name="SO2", fc_name="O2S1", weight=64.066)
species_lib["SH"] = Species_db(name="SH", fc_name="H1S1", weight=33.073)
species_lib["H2"] = Species_db(name="H2", fc_name="H2", weight=2.01588)
species_lib["N2"] = Species_db(name="N2", fc_name="N2", weight=28.0134)
species_lib["SO"] = Species_db(name="SO", fc_name="O1S1", weight=48.0644)
species_lib["OH"] = Species_db(name="OH", fc_name="H1O1", weight=17.007)
species_lib["COS"] = Species_db(name="COS", fc_name="C1O1S1", weight=60.0751)
species_lib["CS"] = Species_db(name="CS", fc_name="C1S1", weight=44.0757)
species_lib["HCHO"] = Species_db(name="HCHO", fc_name="H2C1O1", weight=30.02598)
species_lib["C2H4"] = Species_db(name="C2H4", fc_name="C2H4", weight=28.05316)
species_lib["C2H2"] = Species_db(name="C2H2", fc_name="C2H2", weight=26.04)
species_lib["CH3"] = Species_db(name="CH3", fc_name="C1H3", weight=37.04004)
species_lib["C3H"] = Species_db(name="C3H", fc_name="C3H1", weight=37.04004)
species_lib["C2H"] = Species_db(name="C2H", fc_name="C2H1", weight=25.02934)
species_lib["C2N2"] = Species_db(name="C2N2", fc_name="C2N2", weight=52.0348)
species_lib["C3O2"] = Species_db(name="C3O2", fc_name="C3O2", weight=68.0309)
species_lib["C4N2"] = Species_db(name="C4N2", fc_name="C4N2", weight=76.0562)
species_lib["C3"] = Species_db(name="C3", fc_name="C3", weight=36.0321)
species_lib["S2"] = Species_db(name="S2", fc_name="S2", weight=64.13)
species_lib["S3"] = Species_db(name="S3", fc_name="S3", weight=96.195)
species_lib["S2O"] = Species_db(name="S2O", fc_name="O1S2", weight=80.1294)
species_lib["CS2"] = Species_db(name="CS2", fc_name="C1S2", weight=76.1407)
species_lib["NO2"] = Species_db(name="NO2", fc_name="N1O2", weight=46.0055)
species_lib["N2O"] = Species_db(name="N2O", fc_name="N2O1", weight=44.013)
species_lib["HNO3"] = Species_db(name="HNO3", fc_name="H1N1O3", weight=63.01)
species_lib["SO3"] = Species_db(name="SO3", fc_name="O3S1", weight=80.066)
species_lib["H2SO4"] = Species_db(name="H2SO4", fc_name="H2O4S1", weight=98.0785)
species_lib["TiO"] = Species_db(name="TiO", fc_name="O1Ti1", weight=63.866)
species_lib["TiH"] = Species_db(name="TiH", fc_name="TiH is not included in FastChem...sorry!", weight=48.87)
species_lib["VO"] = Species_db(name="VO", fc_name="O1V1", weight=66.9409)
species_lib["SiO"] = Species_db(name="SiO", fc_name="O1Si1", weight=44.08)
species_lib["AlO"] = Species_db(name="AlO", fc_name="Al1O1", weight=42.98)
species_lib["CaO"] = Species_db(name="CaO", fc_name="Ca1O1", weight=56.0774)
species_lib["PO"] = Species_db(name="PO", fc_name="O1P1", weight=46.97316)
species_lib["SiH"] = Species_db(name="SiH", fc_name="H1Si1", weight=29.09344)
species_lib["CaH"] = Species_db(name="CaH", fc_name="Ca1H1", weight=41.085899)
species_lib["AlH"] = Species_db(name="AlH", fc_name="Al1H1", weight=27.9889)
species_lib["MgH"] = Species_db(name="MgH", fc_name="H1Mg1", weight=25.3129)
species_lib["CrH"] = Species_db(name="CrH", fc_name="Cr1H1", weight=53.0040)
species_lib["NaH"] = Species_db(name="NaH", fc_name="H1Na1", weight=23.99771)

# neutral atoms
species_lib["H"] = Species_db(name="H", fc_name="H", weight=1.007825)
species_lib["He"] = Species_db(name="He", fc_name="He", weight=4.0026)
species_lib["C"] = Species_db(name="C", fc_name="C", weight=12.0096)
species_lib["N"] = Species_db(name="N", fc_name="N", weight=14.007)
species_lib["O"] = Species_db(name="O", fc_name="O", weight=15.999)
species_lib["F"] = Species_db(name="F", fc_name="F", weight=18.9984)
species_lib["Na"] = Species_db(name="Na", fc_name="Na", weight=22.989769)
species_lib["Ne"] = Species_db(name="Ne", fc_name="Ne", weight=20.1797)
species_lib["Ni"] = Species_db(name="Ni", fc_name="Ni", weight=58.6934)
species_lib["Mg"] = Species_db(name="Mg", fc_name="Mg", weight=24.305)
species_lib["Mn"] = Species_db(name="Mn", fc_name="Mn", weight=54.938044)
species_lib["Al"] = Species_db(name="Al", fc_name="Al", weight=26.9815385)
species_lib["Ar"] = Species_db(name="Ar", fc_name="Ar", weight=39.948)
species_lib["Si"] = Species_db(name="Si", fc_name="Si", weight=28.085)
species_lib["P"] = Species_db(name="P", fc_name="P", weight=30.973761998)
species_lib["S"] = Species_db(name="S", fc_name="S", weight=32.06)
species_lib["Cl"] = Species_db(name="Cl", fc_name="Cl", weight=35.45)
species_lib["K"] = Species_db(name="K", fc_name="K", weight=39.0983)
species_lib["Ca"] = Species_db(name="Ca", fc_name="Ca", weight=40.078)
species_lib["Ti"] = Species_db(name="Ti", fc_name="Ti", weight=47.867)
species_lib["V"] = Species_db(name="V", fc_name="V", weight=50.9415)
species_lib["Co"] = Species_db(name="Co", fc_name="Co", weight=58.933194)
species_lib["Cr"] = Species_db(name="Cr", fc_name="Cr", weight=51.9961)
species_lib["Cu"] = Species_db(name="Cu", fc_name="Cu", weight=63.546)
species_lib["Fe"] = Species_db(name="Fe", fc_name="Fe", weight=55.845)
species_lib["Zn"] = Species_db(name="Zn", fc_name="Zn", weight=65.38)

# ions
species_lib["H-_bf"] = Species_db(name="H-_bf", fc_name="H1-", weight=species_lib["H"].weight)
species_lib["H-_ff"] = Species_db(name="H-_ff", fc_name="H&e-", weight=species_lib["H"].weight)
species_lib["He-"] = Species_db(name="He-", fc_name="He&e-", weight=species_lib["He"].weight)
species_lib["H3+"] = Species_db(name="H3+", fc_name="H3+ is not included in FastChem...sorry!", weight=3.02382)
species_lib["HeH+"] = Species_db(name="HeH+", fc_name="HeH+ is not included in FastChem...sorry!", weight=5.01054)
species_lib["Fe+"] = Species_db(name="Fe+", fc_name="Fe1+", weight=55.845)
species_lib["Ti+"] = Species_db(name="Ti+", fc_name="Ti1+", weight=47.867)

# don't forget the electrons! (they may be tiny but they are important)
species_lib["e-"] = Species_db(name="e-", fc_name="e-", weight=5.4858e-4)

# CIA pairs (Note: those come pre-calculated in opacity units (cm2/g) dividing by the weight of the 2nd collision partner in writing order)
species_lib["CIA_H2H2"] = Species_db(name="CIA_H2H2", fc_name="H2&H2", weight=species_lib["H2"].weight)
species_lib["CIA_H2He"] = Species_db(name="CIA_H2He", fc_name="H2&He", weight=species_lib["He"].weight)
species_lib["CIA_CO2CO2"] = Species_db(name="CIA_CO2CO2", fc_name="C1O2&C1O2", weight=species_lib["CO2"].weight)
species_lib["CIA_O2CO2"] = Species_db(name="CIA_O2CO2", fc_name="O2&C1O2", weight=species_lib["CO2"].weight)
species_lib["CIA_O2O2"] = Species_db(name="CIA_O2O2", fc_name="O2&O2", weight=species_lib["O2"].weight)
species_lib["CIA_O2N2"] = Species_db(name="CIA_O2N2", fc_name="O2&N2", weight=species_lib["N2"].weight)
species_lib["CIA_N2N2"] = Species_db(name="CIA_N2N2", fc_name="N2&N2", weight=species_lib["N2"].weight)
species_lib["CIA_N2H2"] = Species_db(name="CIA_N2H2", fc_name="N2&H2", weight=species_lib["H2"].weight)


if __name__ == "__main__":
    print("This module stores information about all kinds of atmospheric species. "
          "Whether they are actually present in exoplanetary atmospheres remains to be seen.")