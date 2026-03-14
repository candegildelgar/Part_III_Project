from matplotlib import pyplot as plt

categories=['Stations inside the caldera', 'Stations outside the caldera']
average_inside_the_caldera_1= 3.6564820480027094
standard_deviation_inside_the_caldera_1= 1.0345932308134784
average_outside_the_caldera_1= 2.1666613901802223
standard_deviation_outside_the_caldera_1= 0.5923108592968364

average_inside_the_caldera_6= 41.22072770480612
standard_deviation_inside_the_caldera_6= 22.441847741582727
average_outside_the_caldera_6= 18.177702916496386
standard_deviation_outside_the_caldera_6= 6.778120144406542

average_inside_the_caldera=0.7681867399386682
standard_deviation_inside_the_caldera=0.06965444473059422
average_outside_the_caldera=0.7025394839860948
standard_deviation_outside_the_caldera=0.09058784614856316

fig, ax = plt.subplots()



ax.errorbar(1, float(average_inside_the_caldera), yerr= standard_deviation_inside_the_caldera, fmt='o', color='red')
ax.errorbar(2, float(average_outside_the_caldera), yerr= standard_deviation_outside_the_caldera, fmt='o', color='blue')



plt.xticks([1,2], categories)
plt.ylabel('Power of the transverse receiver functions')
fig.savefig('/raid2/cg812/Transverse_radial_ratio.png')

