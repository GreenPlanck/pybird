#!/bin/bash
for ell in 0 2 4
do
	cp pz_pk_param.ini tmp.ini
	sed -i "s!ell1 = 0!ell1 = ${ell}!g" tmp.ini
	sed -i "s!ELL  = 0!ELL  = ${ell}!g" tmp.ini
	mv tmp.ini params/pz_pk_param_${ell}.ini
done
