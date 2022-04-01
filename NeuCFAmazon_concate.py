import sys,io
import Relu,Sigmoid

def NeuCFAmazon_concate(x,y):
	W_1=[[-0.15184487, 0.132221, 0.44138223, -0.64995593, 0.94853157, -0.77653158, -0.72724742, -0.77300197, -0.93230313, 0.044084035, 0.23634681, 0.76750487, -0.36192697, 0.45553568, 0.35132143, 0.60457879, -0.39848223, 0.61919194, -0.55890632, -0.54027951, -1.2996453, -1.0383816, 0.46715474, -0.26589739, 0.21552117, 1.1556259, -0.10150944, 0.24446756, 0.57191449, 0.98729914, -0.38764349, 0.42274573],[0.09104877, -0.078568995, 0.70125806, 0.4038181, -0.51051629, -0.50109702, 0.38775209, 0.22231798, -0.32825741, -0.83355242, -0.086307675, -0.32602569, 1.0055664, 0.96444064, -0.56822228, 0.22912472, 0.15377803, -0.39169747, -0.45939976, -0.71841127, 0.28475848, -0.35598683, -0.97706521, 0.20542848, 0.099148132, 0.77220976, 0.86230844, 0.19793233, 0.19720945, 0.20971958, 0.30331635, -1.0094652],[0.091972001, -0.78175217, 0.36057299, -0.27051342, -0.39697823, 0.041893132, -1.1328411, 0.20775287, 0.81270272, 0.013120982, -0.35112867, 0.27343976, 0.95886588, 0.076271549, -0.17914179, -0.57473898, 0.26840147, -0.72068661, -0.42123213, 0.20488432, 0.056296453, 0.22438462, -0.15688039, -0.59459853, 0.754017, 0.44547629, -0.05379751, -0.2679103, 0.2394823, 0.89226705, 0.057567161, 0.50302678],[-0.17303896, 0.33612201, -0.27420822, -0.081891641, -0.26372203, -0.170853, 0.33501795, 0.99199671, 0.26956439, 0.72740614, 0.88297403, -0.24246138, -0.43974236, -0.4220548, -0.72118759, -0.28573516, 0.28548145, -0.67724329, -0.31701678, -0.8256337, 0.56503677, 0.54243964, 0.084165186, 0.41982636, -0.45676082, -0.73720342, -0.13920991, 1.0340552, -0.088591412, 0.054732189, 0.11854146, -0.12521821],[-0.064620882, -0.6643002, -1.1817473, -0.68404567, -0.077840306, 0.74326903, 0.32814887, -0.24677439, -0.75832689, 0.5842486, -0.24039972, 1.0181857, -0.78433675, -0.34436733, -0.038429946, -0.76753926, 0.27110535, 0.077636987, 0.066802338, 0.075169571, 0.53761309, 0.64874005, -0.67149061, 0.38654175, 0.56564289, -0.34235787, 0.059491921, -0.18766113, 0.036662128, 0.09539257, -0.35925671, -0.15252347],[-1.118433, -0.028207123, 1.0820547, 1.6494043, 0.87520456, 0.24932431, 0.3831118, 0.026253872, -0.5190804, 1.2084033, 0.14279111, -0.23731604, 0.60782373, 0.6897909, -0.41393563, 0.031919505, 0.72182435, 0.67032021, -0.23777193, 0.61928028, 0.50705606, 0.096702263, 0.40667447, -0.57039523, 0.37147132, -1.0622252, -0.16068625, -0.19599099, -0.74578869, 0.5057348, -0.091265619, -0.42603946],[-0.73494953, 1.4556059, -0.44183627, -0.84170097, 0.23522818, -0.37285301, 0.52392727, -1.0716258, -0.91584253, -0.29444635, -0.18013267, 0.10848354, -0.22140065, -0.82850677, 0.044982284, -0.1492693, 0.36388358, -0.55982083, 0.23754793, 0.288535, -0.28479949, 0.020254048, -0.14779046, -0.51560599, -0.79723084, 0.51202142, -0.20171344, 0.24013814, 0.38667104, 0.811858, -0.010775528, 0.070088945],[0.27704948, 0.062301353, -0.72058624, -0.079451673, 0.73204666, 1.2272094, -0.33717597, -0.39107087, -0.29038221, -0.51523656, 1.0619951, 0.32156327, -0.062210217, 0.79186243, -0.57278055, 0.12351474, -0.68162501, -0.0015772523, -0.47102159, 1.2340653, -0.59825402, 0.88968533, 0.066918932, -0.19355039, 1.0969033, 0.34966651, 0.49110952, 0.83589709, 0.59181154, -0.8696894, -0.44566658, -0.30231234],[0.22020006, -0.17258196, 1.0132694, -0.057444569, -1.0484082, 0.28056851, -0.72832328, -0.4333013, -0.24433996, -0.31447107, 0.40775275, -0.3230727, 0.53350157, -0.59303159, -0.013633976, 0.075435326, 0.50668293, 0.67027932, 0.62340808, 0.5538848, 0.0032463346, 0.76635939, -0.53643799, 1.2590528, -0.26733696, -0.73188388, 1.0566511, 1.0008755, 0.00057874375, 1.0729686, 0.44588703, 0.60332847],[-0.33698493, 0.70607418, 0.39770669, -0.18949828, -0.045324516, 0.95163667, -0.16906984, -0.3330518, -0.22068603, 0.5639444, -0.67388272, 0.39413217, 0.71829778, -0.21937318, 0.023331333, 0.16528377, 0.18970266, -0.19624302, -0.92701298, -0.40406087, -0.47734725, -0.08172477, -0.072367698, 0.16199301, 0.42660341, 0.16818625, 0.37110639, 0.24698377, -0.67965096, -1.0533595, -0.5683803, -0.18075731],[0.86162412, 0.15064713, -0.075439222, -0.35260811, 0.4826481, 0.22469534, -0.69282925, -0.0098513104, -0.20785984, -0.10891578, 1.0124608, 0.049520295, 0.50059348, 0.46485689, 0.78579146, 0.30699775, 0.10979292, -0.83504546, -0.77376431, 0.34560567, 0.81789243, 0.41059738, -0.45404932, 0.26200935, -0.20010327, 0.35195851, 0.15782258, -1.0022571, 0.079934061, 0.58052021, 0.08342997, 0.021863898],[-0.14396317, -0.55329758, -0.52500641, 0.32826003, -0.92901665, 0.13338885, 0.47733667, -0.72950357, -0.38879538, -0.021537393, -0.39164072, 0.39525989, -0.6903016, 0.17956731, 0.11399478, 0.26447025, 0.37138972, -0.36177641, 0.28016233, 0.5530588, 0.15645537, -0.68177277, 1.2145079, -1.0879322, -0.08860188, -0.77058071, 1.2249618, -0.18547085, -0.61037922, -0.23772742, 0.85405105, 0.035211042],[0.30514592, -0.755072, -0.46175036, -1.3159883, 0.21698995, 0.66310263, -1.4658774, -0.27393177, -1.1285123, -0.27965376, 0.055161938, -1.5191687, -0.033943985, -0.31218755, -0.98613453, -0.22542225, -1.1331069, -0.46928906, -0.39721227, -0.022038059, 0.25047073, -1.2955474, -0.68286496, -0.81472731, 0.099845536, 0.12067533, -0.75124407, 0.1875447, -0.1419723, -0.50310802, -0.12254091, 0.59719956],[0.68473804, -0.32591525, 0.36017147, -0.57388979, 0.31852832, -0.37636784, 0.077652298, -0.1096936, -0.44970155, 0.76297438, -0.56045544, 0.58470666, -0.15758893, 0.30748689, -0.03397207, 1.1755799, 0.019331194, -0.025026225, -0.82354641, 0.25911283, -0.21005233, -0.13144961, 0.66093028, 0.79712462, 0.18501338, -0.73245126, -0.491606, 0.88695115, 0.72591609, 0.52933306, 0.18933703, 0.29876608],[-0.47362643, -0.27072108, 0.56003439, 0.077179812, -0.15163714, -0.014091765, 0.27918142, -1.0364914, 0.31047928, 0.83979863, 0.17917433, 0.23884553, -0.87148654, -0.28845721, -0.06618198, -0.54598963, -0.53636461, 0.68327278, 0.11049772, 0.24786699, 0.9834246, 0.18148583, -0.009402086, 0.39262193, 0.42106968, 0.99398208, 0.94480073, 0.75305057, -0.30865675, 0.063307628, -0.23351777, -0.2770974],[-0.80865139, -1.4065062, 0.37415764, 0.0027431846, 0.78648633, 0.45992148, -0.1924637, -0.60366726, -0.06526988, -0.71001667, 0.47449234, -0.13500783, -0.64990991, -0.5243842, 0.21998534, 0.76767558, 0.037768938, -0.60646802, -0.47029713, -0.50713259, -1.0191323, -0.68453389, -0.12246263, 0.563362, 0.16772605, 0.073575705, -0.29027188, -0.3809278, 0.38039207, 0.51482332, 0.11180832, -0.32252163],[0.7879681, 0.5670653, -0.23274988, 0.48994902, -0.8426792, 0.94553632, 0.040784258, -0.57388753, -0.096990578, 0.25363719, 0.92101461, -0.56270027, -0.30486903, 1.1213982, 1.2658383, -0.54055065, 0.16245313, -0.0085225934, 1.0572444, -1.1735107, -0.51287729, 0.43503928, 0.093757965, -0.22300689, 0.34216017, 0.20285498, 0.80028188, -0.58284122, 1.5833334, 0.82590711, -1.0151867, 0.83311462],[-0.81473947, 0.18975937, -0.64401996, -0.70729929, 0.1542384, -0.39839992, 0.56115937, 0.072088577, 0.19545305, 0.084817298, -0.3458221, -1.3137758, 0.64886802, -0.093060724, 0.506212, 0.16704983, 0.26228479, 0.32417357, -0.3018541, 0.14167589, -0.61682671, 0.21257597, 1.1451095, 0.58644474, 0.43636298, -0.21643715, 0.39568394, -0.90696919, -0.25630775, -0.49454099, 1.3119118, -0.72984892],[-0.087946303, -0.028273342, 0.35963863, 0.072161838, -0.142033, 0.38651326, -0.85902286, 0.49460176, 0.45875084, 0.4646816, -0.75164896, -0.68081731, -0.47308826, -0.53270882, -0.40864488, -0.92476481, -0.73192203, -1.184269, -0.10784438, 0.42820227, -0.20729147, -0.32221779, 0.073558018, 1.3686181, -0.38667476, 0.27773181, 0.097533219, -0.9772588, -0.14340678, 0.73208082, -0.55446744, -0.37313643],[-1.0930363, 1.3167057, 0.16216244, 0.23366432, -0.39644399, -0.32922989, -0.35434034, 0.60860533, -0.11878362, -0.56864762, -0.15981473, -0.87079269, -0.60518986, -0.27604756, -0.18073784, 0.32368696, -0.61646098, -0.28899732, 0.20059212, -0.054604772, 1.0298463, -0.20682813, -0.43098336, 0.43978095, 0.74427742, -0.018334772, 0.22223304, 0.14409274, -0.64926022, -0.018614573, 0.28432643, -0.23182017],[-0.45921582, -0.17471731, -0.6343174, 0.40875679, -0.94123077, -0.57964933, 0.80139428, -0.79780436, -0.87554508, 0.24859157, -0.12358831, -0.48086038, -0.55107623, -0.79482388, 0.070885092, -0.82229549, -0.027352253, -0.040454246, -0.79348302, -0.11403947, -1.3979826, -0.0069569084, 0.11857136, -0.6148625, -0.37889582, 0.50025922, -0.89797962, -0.76892579, -0.69610858, -0.1535465, 0.47987691, 0.044355992],[-0.28398934, -0.26091602, -0.042045861, -0.36321542, 0.38483021, 1.3676871, 0.16482025, 0.99507928, -0.30519298, -0.15028608, -0.12346987, -0.25235301, 0.70953006, -0.471553, 0.70211905, 0.38097414, -0.72334254, 0.54995126, -0.76627308, 0.66608799, 0.84725767, -0.25421354, 0.74170607, 0.14061208, 0.13311535, 0.40913194, 0.79377794, 0.66714466, 0.0051692822, 0.96453887, 0.6931268, 0.77235484],[-0.080112197, 0.6853959, -0.42964271, -0.30928588, -0.81748849, -0.12360453, 0.11580212, 0.11273391, 0.38346338, 0.10309164, 0.35475683, 0.45976964, 0.95244235, 0.18632829, -0.31297567, -1.0252739, -0.95206785, -0.22935258, 0.64877009, 0.011518489, -0.53005344, -0.67993146, -0.54157674, -0.10453649, 0.39493132, -0.79685235, -0.036578476, 0.15486066, -0.14592691, 0.29648852, -0.31275696, -0.61671335],[-0.97992563, 0.25543365, 0.16783015, -1.0289932, -0.37189242, -0.17847808, -0.14482084, 0.036913373, 0.82384861, -1.6302016, 0.18335061, 0.37849244, 0.20052524, 0.68146956, 0.2471016, -0.53427178, 0.32972705, -0.14275274, -0.50687516, 0.64166266, 0.32628196, -0.30932066, 0.96738142, -0.56607771, -0.25987157, -0.76781684, -1.111867, 0.26854891, -0.58979774, -0.19648574, -0.80700892, -0.19355559],[0.024473118, -0.23994195, -0.37040681, -0.1001942, -0.39735124, 0.58062911, -0.65734255, -0.45957717, -0.011071592, 0.30166155, -0.29377502, 0.13477446, 0.41470733, -0.20376438, 0.45404983, 0.20805378, -0.92312908, -0.33370611, 0.24728858, 0.086526543, 0.78093952, 0.67290848, -0.52432889, -0.67480779, 0.41804034, -0.11999523, -0.26213256, 0.91947454, 0.32591739, 0.3703787, -0.98442805, -0.89743721],[0.63961309, 0.98457068, 0.059997082, -0.1886062, 0.48215687, 0.62784791, -0.37246999, 0.12896287, 0.16089281, -0.27239308, -0.65761548, -0.2854577, -0.88889074, 0.48535046, 0.4642944, 0.074098982, 0.29169253, 0.85655361, -1.2402499, -0.28000823, -0.39514086, -0.67152387, -0.35701299, -0.4649305, -0.41087624, -0.274232, 0.13161404, 0.4391486, 0.62633955, 0.37475184, 0.21176876, -0.95431656],[0.016527489, -0.86710203, -1.1275487, 0.24325626, 0.57768649, -0.42742732, 0.41085777, -0.10075901, -0.097697444, -0.81677383, -0.18415011, -1.0700673, 0.38429689, -0.81384772, -0.33807805, -0.72120601, 0.32991758, 1.0951226, -0.31326517, -0.51173019, -0.0085785165, -0.27956411, 0.82066804, 0.548195, -0.64735317, 0.18978439, 0.32027179, -0.17758028, 0.079870321, 0.44327483, -1.0832642, 0.32188606],[-0.98120409, 0.19663207, -0.67311114, 0.64431441, -0.5962643, 0.27890348, -0.86542714, -0.1574854, 0.63814521, 0.56328183, -0.002258474, 0.74866921, 0.30916706, 0.27180073, -0.18161406, 0.1269322, -0.24209659, -0.20388903, -0.61773866, -0.1313533, 0.60202473, -0.73336077, -0.79338872, 0.6010589, -0.63683838, 0.027687054, 0.71447927, -0.20971736, -0.21968606, -0.73682404, 0.37525046, 0.088348404],[-0.035048213, -0.17519069, -0.75698799, -0.44743946, -0.53207171, 0.070243716, -0.11105284, 0.83906215, -1.2704936, 0.60063738, 0.24586163, -0.056616142, 0.11703339, 0.78893214, 0.54241216, -0.32874936, 1.7371134, 0.14951415, 0.34041235, 0.41458526, 0.61620075, 0.14700612, -1.0978931, -0.45645168, 1.1114149, -0.0077596316, 0.12441038, 1.031279, 0.98174673, -0.57986063, 0.28462842, -0.80514151],[-0.6103543, -0.35048863, -0.71261281, 0.34129715, 0.56432146, 0.6577062, -0.3192361, -0.24976234, 0.78501207, 0.4327974, 0.42175379, 0.32605469, -0.1740334, 1.2413001, -0.16021904, 0.45842424, 0.27441907, -0.82544583, 0.76102006, -0.44652858, -1.1295315, 0.28364184, -1.0322649, 0.035468362, -0.12984167, -0.11775161, -0.40387172, 0.51356518, -0.9845233, 0.36238006, 0.066761814, 0.23377064],[1.1235114, 0.39584059, -0.18372819, -0.30360416, -0.067464389, -0.44204843, 0.50766206, 0.25368688, 0.78655863, 0.48036733, 0.36947525, -0.38853097, -0.3047021, -0.30709818, -0.16138618, 0.63440466, 0.67623585, -0.21197475, -0.091313712, 1.4276893, 0.66346943, 0.024738159, -0.75836962, 0.22004203, 1.5986993, -0.39258438, 0.41869038, 0.76040483, 0.77375603, 0.6627028, -1.074075, 0.027751023],[-0.10412849, 0.18069765, 0.046895295, 0.74521077, 1.0305614, -0.89943367, -0.034670178, -0.20869099, -1.0114312, -0.38338929, -0.084013321, 0.1057279, -0.20925781, -0.082015619, 1.391965, -0.41316715, -0.19636048, -0.11475603, -0.79650182, 0.50854135, 0.27401167, 0.25194868, 0.51044983, -0.001054083, 0.74694508, -0.85926837, -0.010923053, 0.85742444, -0.35206434, -0.21410225, 0.44215247, -0.25722271],[-1.5078604, -0.20148565, 0.1615769, 0.66226554, -1.7287099, -0.84486598, -1.4824494, 0.12037022, -0.14406641, -1.4938116, -0.035355058, -0.19939269, -0.81294745, -0.33727241, 0.042202879, -0.89585018, -0.2979123, -0.70307016, 0.12039325, -0.19244693, -1.0286293, -0.3334955, -0.50077981, -0.78839493, -0.38234401, 0.20461629, -1.364282, 0.71661353, -0.48497006, -0.018194705, 0.12420423, 0.035281364],[0.06326399, 0.74266124, 0.24859586, 0.28538468, -0.74105179, 0.64917356, -0.13941602, 0.50750417, 1.6305653, 1.1865699, -0.28649494, 0.6024732, -0.72008073, 0.59619194, 0.14141399, -0.22922781, -0.5774256, -0.050546158, -0.28826505, -0.80675375, -0.64932764, -0.40428644, -0.17143661, 0.693398, -0.89344794, -0.16043562, 0.27584371, -0.24330528, -0.1484009, 0.8696115, 0.26025099, 0.012178341],[0.52150065, 0.49347645, -0.075816706, 0.57489002, -0.31743196, 0.76295972, 0.71268904, -0.37425798, -0.70610052, 0.13352142, 0.28088027, -0.14997727, -0.69703645, 0.75481319, 0.60936183, -0.51167929, 0.68059337, 0.23592848, -0.73436189, 0.21423025, -1.2910498, 0.6574778, 0.16575067, 0.3894859, 0.11055604, 0.51917517, 0.13376798, 0.49140063, 0.0059072361, 0.53319752, -0.20742346, 0.65308255],[-1.2302101, 0.18495455, 0.64022487, 0.32801506, -0.73461008, -0.14726606, 0.35820001, 1.0176334, -0.31297264, 1.9173487, 0.42859223, 0.59665143, -0.22380663, -1.2035277, -0.42431885, -0.22990106, 0.70706296, 0.25780618, 0.038958531, -0.30257541, 0.48627609, 0.3252306, 0.9262498, -1.0118439, 0.37814727, 0.037863668, 0.63268489, 1.0758846, 0.28179321, 0.79845107, 0.83398038, 0.57242709],[0.59002173, 0.2519792, 0.68249679, -0.43615982, -0.25608376, -0.51959968, -1.1390564, -0.68670577, -0.280931, -0.43453065, -0.067903489, 0.1551118, 1.1001068, 0.52802616, 0.3089464, -0.078599624, -0.28112018, -0.42736769, -1.4167057, -0.0089289853, -0.1287708, 0.0015757058, 0.43426144, 0.76604879, 0.13251202, -0.57153928, -0.17788444, -1.5357628, 0.50314808, -0.2395073, 0.35633779, -0.41773728],[-0.19220728, 1.3378623, -0.53834629, 0.826868, -0.31467327, -1.2751821, -0.040511828, -0.13778707, 0.35355172, 0.14040449, -0.46744633, -1.2101775, -0.040310316, -0.26594552, -0.070503168, -0.58443868, -0.70732784, 0.20392364, -0.48677382, 0.80811453, 0.66331059, 0.82336515, -0.22739635, -0.39122808, 0.17622356, 1.014393, 0.25670508, -0.073462531, 0.25577137, 0.83390003, 0.48616719, -0.790075],[0.02517291, 0.1467161, -1.3974143, 0.057982903, -0.36303607, 0.80769593, 0.70393735, 0.17028087, 0.13419293, 0.30514479, 0.3501744, 0.6929273, 0.059553429, 0.30792788, 0.21993987, -0.51419932, -0.77958411, 0.51160848, -0.59417152, -0.25138059, 1.3170583, 0.420834, 0.15831737, -1.0466615, -0.23663035, 0.99696785, 0.58465958, -0.044824947, 0.031807121, -0.91437835, -0.075021259, 0.42621967],[1.4741577, -0.58367372, 0.40756178, 0.77941608, -1.0344734, -0.75090814, 1.0770613, -0.014909409, -0.52077216, 0.034391809, -0.039923139, 0.46818119, 0.51537442, 0.540672, -0.69338655, -1.034747, -1.3645289, 0.66428131, -0.8659066, 0.20527212, 0.13331461, 0.90912575, -0.90409333, 0.095633224, 0.33346564, 0.93140388, 0.19616054, 2.7871091, -1.0084267, 1.5746258, -0.4610669, 1.1807586],[-0.49053612, -0.016395045, -0.40511307, -0.38753322, 0.065401092, -0.99915689, 1.4870626, -0.59848964, -0.077005126, -0.30900308, -0.30816093, 0.020927809, 0.074090734, -1.2320534, -0.33146548, -0.066970363, -0.64264137, 0.29575524, 0.60200721, -0.93291622, -1.3567742, -0.12754568, 0.80932575, 0.014281577, -0.51034057, 0.39847425, 0.46719778, -1.0907047, -0.58565933, -0.10268907, 0.86353332, -0.88655543],[0.57680809, 1.3166904, 0.36307836, -0.198183, -0.53947532, -0.16430575, 1.1073555, -0.30786011, 0.96441525, 0.45266086, 0.021556024, -0.52347863, 0.14211431, 0.31364048, -0.37482485, 1.5557835, 0.22005847, 0.89356422, 0.26307324, -0.60769522, 0.21072935, 0.026789386, 0.78572613, 0.068529613, -0.37874064, -0.12066493, 0.66266942, 0.42553622, 0.79773206, -0.66221946, -0.22224867, -0.53223473],[-0.66113514, -0.46403575, -0.52217323, -0.042585548, 0.4338803, -0.063576594, -0.44232461, -0.10796694, -0.44606644, 0.35020059, -0.61060196, -0.0074013742, 0.28705111, -0.13486302, -0.66633743, 0.90005863, 0.29527336, -0.38092774, -0.55811697, -0.18185489, -0.34528509, -0.28199661, -0.67682081, 0.28329733, 0.26555935, -0.74927884, -0.59713954, 0.99591643, -0.41920078, -0.46932304, -0.11184939, -0.42391488],[0.13824058, -0.89343762, 0.48923606, -0.25568104, -0.4555141, 0.19358048, -1.2712017, -0.064638883, -0.1910843, -0.098134145, 0.031313118, -0.017022416, 0.37212431, 1.4221969, 0.019104645, 0.17918077, -0.46611956, 0.57483697, 0.3920975, 0.72768581, 0.40350196, -0.98393291, -0.16767509, 1.2791352, 0.93219972, 0.71958786, 2.2800827, 0.058498371, 1.4494289, -0.055426963, 1.6760836, -0.68163925],[-1.1126969, 0.61653274, -0.28668967, -0.27180776, -1.0101539, 0.61749047, 0.13838696, 0.65307373, 0.28488481, -0.45635626, -0.75864136, -1.2985132, 0.72779286, -0.81248325, -0.37562573, -0.86854106, 0.6364159, 0.25837764, -0.82865846, -0.95229548, 0.45142177, -0.88615149, 0.56648546, 0.51396751, -0.99000293, -0.041596428, 1.1299192, 0.019077742, 0.97411001, 0.3185378, 0.052690364, -0.74124748],[0.51185083, 0.12921537, -0.043680236, 0.53923458, 0.15765774, -0.58902073, -0.49456847, -0.55465579, 0.82310504, -0.7935164, 0.7035498, -0.80631363, 0.16986489, -0.32570863, -0.76375222, -0.18754773, 0.43077844, -0.82729656, 0.58191818, 1.1287839, 0.18041077, 0.52963507, -1.2092466, 0.60080642, 0.6035102, -0.79206234, 0.74192441, 0.41339061, -0.22792757, -0.47548419, 0.99881476, 0.54289776],[0.76398593, 0.03364962, -0.35354769, -0.55859864, -0.27556971, 0.025844734, -0.2534585, -0.37376601, -0.27003935, 0.42344311, -0.17370433, -0.057153959, -0.97058672, 0.21997978, -0.33646739, -0.65104318, 0.3771714, 0.94408011, 0.82572031, -0.31472448, -0.57153338, -0.75647008, 0.18924169, -0.75799257, -0.08354874, -1.0409108, 0.12698098, 0.1433083, 0.28396061, 0.067952663, -0.20339687, -0.22547238],[0.60554266, -0.20231102, 0.24956715, -0.29270223, -0.72611773, -1.3843744, -0.27433527, 0.93555498, 0.083351344, 1.1627938, -0.46602249, -0.33788055, -0.22533682, -0.019947244, 0.11675458, -1.103768, 0.95245659, -0.66984785, 0.71804738, 0.28943497, 0.13488764, -0.31980807, -0.19179411, 0.23548266, 0.80490053, 0.94900215, 0.3755517, -0.89752626, -0.7298559, 0.041216616, -1.3052272, 0.14960542],[-0.12522307, -0.50857633, -0.82505929, -1.1042163, 0.11222862, 0.62356412, 0.24211866, -0.0089253569, 0.79322261, 0.21429589, -0.24125479, -0.0076339855, -0.38107303, -0.25082114, 0.14358391, -0.30246684, 0.13319507, -0.74567509, 0.52667826, -0.46161321, -0.11839792, 0.47826511, 0.6686219, 0.21978815, -0.34141549, -0.26876885, -1.188197, 0.02206271, 1.0382797, -0.021137092, 0.22471483, -0.081793673],[-0.26526585, -0.99959266, -0.34712601, -0.18729267, -0.3617802, -0.075426951, 0.53155029, -0.91459554, 0.86431581, -0.1652863, -1.428133, -0.72849762, -0.70417285, -1.0122665, 0.096645668, -0.37380636, -0.019449176, 0.22574805, -0.40679914, 0.67836988, -0.12740922, -1.5129988, 0.92181504, -0.26135451, 0.53652048, -0.6468972, -0.044538863, 0.093941055, -0.9790442, -0.021588508, -0.7493813, 0.53235096],[-0.67663896, -0.0089985449, 0.93061471, 1.5569607, -0.25147432, -0.19432981, 0.5736714, -0.267147, -0.81835943, 0.0058141137, 0.12321834, -0.1702407, 0.16663897, -0.4897081, 0.094630726, 0.46961486, 0.25366819, -0.053122535, 0.3367767, -0.51413298, 1.2574357, -0.57054007, -0.65643579, 0.33278912, 0.0017501916, -0.6643768, -0.19931203, -0.59764546, 0.70028377, 0.5257529, 0.54352409, 0.15789294],[-0.81817573, 0.93150645, -0.72464371, -0.86437392, -1.2771225, -0.28619355, 0.52638662, -1.3112074, -0.28387278, -0.82087994, 0.37642035, -0.50165361, 0.91670036, 0.32951254, -0.32295561, -0.014407261, -0.53974897, -1.1379886, -0.14681542, 0.031996168, -0.23804477, -1.4554917, -0.14221986, -0.22601946, 0.31761947, 0.069901459, 0.09922082, 0.57966501, 0.058512181, 1.1165764, -0.44525707, 1.276915],[0.29337299, -0.055626661, 0.62043464, 0.33380046, -0.66717917, 1.2309674, 0.68597186, -0.97748083, 0.12700485, 0.19298434, 0.24076636, -0.14730439, 0.71673846, -1.0694497, -0.061053522, 0.17617597, -0.53546584, -0.28167826, 0.17650771, -0.095422208, 0.021907492, 1.0124542, -1.0210326, -0.33286625, 1.0393339, 0.3620066, 1.050568, -0.1381277, 0.59774196, 0.72509724, -0.79363102, -0.42743951],[-0.18903889, 0.59885424, -0.68774503, 0.027639933, 0.34725234, -0.39470586, 0.44402334, 0.32524103, 0.64438719, 0.27652094, -0.30839044, 2.2055609, -0.64085382, -0.45460927, -0.15189472, 0.65369475, 0.6860761, -0.15344773, -0.53350967, 0.87657362, 0.21122682, -0.71510035, -0.97973382, 0.052548006, -0.035287213, 0.34718829, 0.91595125, 0.050535511, 1.192158, 0.44101322, 0.27934119, -0.2104909],[0.12220638, -0.23131046, 0.41083726, 0.68245226, -0.25836089, 0.26051947, 0.15407944, 0.42899713, 0.84157085, 0.22360955, 0.094752192, -0.14721408, -0.52734381, -0.18126716, -2.3727808, 0.12157328, -0.62727022, -1.0065314, -0.64196569, 0.22833563, -0.30412906, -0.36566129, 0.020698611, 0.084683746, -0.021434005, 0.95070589, -0.88567477, -0.55185813, 0.45875478, -0.35648298, -1.3206246, 0.45395097],[0.34429532, -0.94743693, -0.46706152, 0.59450066, -0.44139877, -0.65827161, 0.23969796, -0.65423566, -0.34575969, 0.26982933, -0.48653454, -0.11300071, 1.4564637, -0.24754128, 0.20447153, -0.20430051, 0.77275133, 0.16326295, 0.07243944, -1.1013901, -1.3175495, -0.77568769, 0.76106453, -1.5561768, -1.4246644, -0.010564037, -0.26002616, -0.77817249, -0.50299722, -0.30201131, 0.33572361, 0.67876643],[-0.054660089, 0.49577323, -0.53829652, -0.60817951, -1.3683513, -0.18184936, 0.75050294, -1.084849, -1.2316314, 0.22121808, -0.99131173, -0.24056524, -1.0377346, -1.6831038, 0.22279644, -0.69845641, 0.58349079, -0.40253782, -0.68996245, 1.0990967, -0.051522858, -0.64299017, -1.8666383, 0.045917459, 0.029739166, 1.2860234, -0.42195606, -0.38087496, -0.83379781, -1.2756206, -0.013175397, 0.38066158],[-0.18558602, 1.0067945, -0.57922691, 0.30542088, 0.1209651, -0.81965548, 0.027008075, -0.88769066, -0.4245359, 0.49381724, 0.010749256, 0.47891241, 0.059277106, 0.025429463, 0.27025756, -0.97279209, -0.40676865, -0.33224025, -0.14010628, -0.36628446, -0.49221402, -0.45705864, 0.63656366, -0.017692847, 0.38460469, -1.0848209, -0.067686886, 0.21711186, 0.72445703, -1.1092551, -0.49831501, -0.71340156],[1.0186331, -0.60129279, -0.058547072, -0.48230982, 0.32899129, -0.20845747, 1.0230125, 0.29162586, -0.62131256, -0.70476872, 0.34265548, -0.7857362, -0.69439292, -0.24051507, 0.33265507, 0.56889516, -0.12473068, -0.70430869, -0.94379073, -0.41088173, -0.010678188, -0.73990816, 0.32161501, -0.88636607, 0.094204694, -0.84226686, 0.39628163, -0.070914827, -0.84219396, 0.012981794, 0.90747255, -0.36149251],[0.54708111, 1.0102844, -0.090853207, -0.2868492, 0.89127344, 0.821666, -0.095371358, 0.52438986, -0.65501767, 0.31992108, -1.1849357, -0.50102228, 0.29724556, 0.73261243, 0.44638184, -0.20510955, -1.0384167, -0.43635505, 0.46079022, -0.29339519, -0.34564304, -0.71380991, -0.11212889, -0.65141088, 0.64900243, -0.16405725, -0.31814823, 0.3664557, -0.07415203, -0.69129914, 0.44599482, 0.15049945],[-0.6842165, -0.63766402, 0.48939085, 1.1210783, -1.4083787, 0.50619519, 0.9895162, 0.0014227618, 0.27311724, -0.76338387, 0.12858772, 0.25837478, -0.073284924, 0.21468575, -0.24058266, -1.0077534, -0.13406853, 0.065176047, -0.33518299, -0.30515552, -0.15332039, 0.35776573, 0.34680825, 1.6067072, 0.35268432, -0.49668947, -0.38370892, -0.16565666, 0.61928302, -0.16247934, -0.017415887, 0.019351013],[0.53630555, 0.26993859, -0.53283495, -0.0074199517, -0.42725134, 0.28582495, -0.41031098, 0.4510105, -0.90183407, -0.096136801, -0.48262861, 0.96149826, 0.1544608, 0.45636609, 0.75896299, 0.21210605, -0.45533907, -0.73867923, 0.64859521, 0.038898025, 0.72326541, 0.75572038, 0.57116675, 0.65343267, -0.89088541, -0.36947018, 0.62827384, 0.6409592, -0.115678, 0.23067433, -0.23455456, 0.41388541],[-0.066377178, -0.22146297, -0.32572824, -0.48305562, 0.7450698, -1.3062365, -1.031628, -1.063723, -0.65051013, 0.68257642, 0.62754852, -0.99931115, -0.57400882, -0.89353174, -0.39831549, -0.059511598, -0.40884417, 0.53745478, -0.40515533, -0.44388995, 0.13770971, -0.81947106, -0.22041988, 0.75334185, -0.85938621, -0.34838057, -0.80883813, 0.11495037, -0.26715127, -0.14551723, -0.22535624, 0.51189268],[-0.1316296, -0.23860772, 0.62218165, -0.67159414, 0.45478949, -0.68129385, -0.055352613, -0.80177438, 0.189358, 0.21748874, -0.15506211, -0.12426902, -0.35552588, -0.03574587, -0.44391093, -0.51789141, 0.59560049, -0.40344369, -0.3760975, -0.031370383, 1.0139357, 0.078877151, -0.10903253, -0.016463807, 0.19633856, -0.37289789, -0.022069018, -0.063695721, -0.18527661, -0.1978105, -0.054122496, -0.92347556]]
	b_1=[-0.726912,-0.468963,-1.84458,-2.50053,2.18258,0.293291,0.195877,-1.34544,-0.904689,-0.0344578,-2.78136,-1.34331,-1.98254,0.174777,-2.43852,-2.88546,-1.12131,-2.39827,-2.81123,-2.10309,0.0600595,-0.968074,-0.536563,-0.548613,1.24619,0.7046,0.270467,0.172235,1.77475,-2.19497,-1.37764,1.63555]
	W_2=[[-0.14926946, 0.89975566, 0.57127053, 0.91070926, 0.53452212, 0.05977251, -0.46566677, 0.022921348, 0.14750829, 0.23466286, -0.44648468, -0.62313879, 0.1028042, 0.087053694, 0.036583871, -0.21390626],[0.89201683, 0.0012735837, 0.93002093, 0.40590018, 0.472361, 0.51314896, 0.41934979, -0.15891038, -0.48227414, -0.55100924, 0.65030414, -0.17380486, 0.80966836, 0.07051456, 0.7803424, 0.84769309],[-0.27871993, 0.12400093, 0.17743757, -0.32860836, -0.4764685, -1.9642085, 0.71629727, 0.41391897, 0.62416464, -0.77084333, -0.057586409, -0.19237635, -0.37389809, -0.16606252, 0.28808185, 0.25295478],[0.58431488, 0.18757783, 0.077695176, 0.6270436, 0.66032135, -1.3581151, -0.46470866, -3.0241292, -0.52491891, -0.27640614, -0.38463885, 0.25099674, 0.62601393, -0.575297, 0.013767883, 0.18963048],[-0.45872888, -2.4913764, 0.56672812, -1.3715791, -2.3512249, 0.42986277, 0.0056631807, 0.19423872, -0.25653195, -1.3030206, -0.24200435, -0.70410728, -0.028407535, -0.51524395, -0.105069, -0.14153905],[0.3271445, 0.7470848, 0.25047669, 0.71130908, 0.34337661, -0.047436394, -0.65882444, 0.1370665, 0.091352709, 0.30372524, -2.2629008, -3.0975068, 0.36675549, 0.34368339, 0.26912507, 0.57303113],[-1.132883, -1.4316224, -0.38948834, -0.22666031, -1.2753465, -0.60427189, 0.16960175, 0.048351899, 0.56875139, -1.0498209, -0.15792958, -0.43677866, -0.42094091, 0.32230344, -0.064857163, -0.17792207],[-0.5398401, -2.0899651, -0.60630012, -0.099577792, -0.3258003, 0.036428202, 0.3639864, -0.23899966, 0.6828447, -0.36285272, 0.25696686, -0.23242514, -0.67055511, 0.26417288, -0.57063848, -0.044355717],[0.71229964, -0.29087174, 0.72278011, -0.16859101, 0.21350761, 0.41240501, 0.27270937, 0.92196524, 0.028075816, -0.077695921, -0.92063379, 0.10232408, -0.1051117, -0.050353963, 0.31142819, -1.8711476],[0.17648149, 0.6209718, 0.48767942, -0.10875709, -0.42021421, -0.28990293, 1.0941123, -0.008159874, 0.7660448, -0.16638695, 0.14327636, 0.084302388, -0.81304121, -0.2974827, 0.34860814, -0.31161717],[0.82225704, 0.1092257, -0.31596661, 1.4135787, 1.842589, -0.094723716, -0.54609692, -0.6787833, -0.62960768, -0.64143616, 0.29985169, -0.32727337, 1.0361191, -0.031204145, -0.34272873, -0.8094511],[-0.27678356, 0.69603425, -0.25669721, 0.9824217, 0.65824229, 0.54685557, 0.22366379, -0.88294756, 0.36781052, -1.0123991, -0.73261696, -2.0245678, 0.058908354, 1.1076651, 0.25763217, 0.117817],[1.6962429, 0.93496138, 0.8066498, 0.89362872, 0.91063297, 0.32512161, -0.58544129, -1.7156011, -0.4126792, -0.17783418, -1.3658894, 0.47520131, 1.7354231, -1.9771332, -0.065939605, 0.48590088],[-0.28402114, -0.68092638, -0.45802379, -0.90401256, -1.1547066, 0.60320663, 0.24473511, -0.070457667, 0.41076881, 0.23834597, 0.35502851, 0.086198963, 0.14864324, 0.19042192, -0.39560124, -0.088808842],[0.35773751, 0.19636512, 1.21259, 0.23136012, 0.19593495, -0.31967854, 1.1194844, 0.89540857, 0.30711186, 0.33962548, -0.37365815, 0.46606612, 0.085409127, 0.00021218862, 0.76323295, -0.26989478],[0.79293686, 0.41871598, -0.81003857, 0.93736917, 0.77648753, -2.0883105, -0.87610209, 0.64630741, 0.34241733, 1.0028009, -0.85695213, 0.54376465, 0.75523382, 0.2906242, -0.079017945, 0.1631417],[-0.34297034, 0.72062987, 0.72820187, 0.5476284, 0.057708941, 0.88834596, -0.046840794, 0.33313876, 0.0060301246, 0.65116465, 0.69657815, -0.48789302, 0.13991746, -1.0026455, 0.55007511, -0.39096421],[-0.25535041, 0.81140745, -0.46664074, 0.95163441, 0.56682432, 0.88538629, 0.76702666, -0.23664702, 0.33781695, 0.61532986, -0.072195575, 0.57629496, 0.62605, 0.55295819, -0.40350759, -1.0221022],[0.08499518, 1.0905764, 0.29573253, 0.98714662, 0.55421835, -0.052066371, -0.75114715, 0.66056371, 1.0125624, 0.6791355, 0.081551023, -0.015737485, 0.63594174, 0.07788834, 1.0042003, 1.5338328],[-1.0481838, -1.4185771, -0.39354864, -0.27147156, -0.73826718, -1.5429862, 0.16878265, 0.17623495, -0.039218411, -0.18727468, -0.37935901, 0.22545551, -0.93178087, -0.86573613, -0.84119248, -0.47637156],[-0.16696069, 1.3085407, 0.045801766, 0.67197996, 0.64922327, 0.5330475, -0.79452586, -0.13548875, 0.33038992, 0.56543112, 0.95358336, -2.357312, 0.83853781, 0.43502718, 0.6036253, 0.93148088],[0.12826194, -0.071821712, -0.071351416, -0.73479092, 0.55896968, -0.1653496, -0.17277056, 0.39410326, -1.4121019, -0.17683481, 0.34557912, 0.91118562, -0.52115291, -0.28424183, -0.72365755, -0.12399998],[0.35545459, 0.62507349, 1.3054669, -0.15788907, 0.094384551, -0.70707327, -0.8228761, -0.076384328, 0.36375451, 0.13936035, -0.62098652, 0.41537499, 0.25930303, -0.75834209, 0.4012931, 0.64577681],[-0.4396975, 1.0319563, -0.020516552, 0.89869738, 0.50823712, 0.37268171, -0.0928187, -0.58658284, 0.42278731, -0.18455826, -0.90722215, -0.1734378, 0.5066905, 0.99650788, 0.16794926, 0.46892247],[0.21944353, 0.17445377, 0.14094067, 0.52356142, 0.68281865, 0.090594634, -0.11888552, -0.6454432, 0.086172417, -0.6226235, -0.8297177, -1.2795312, 0.82489753, -0.60672241, 0.1828607, 0.8124671],[0.52644563, 0.78747445, 0.39133036, -0.025515513, -0.097591095, 0.1254364, -1.3015372, 0.52116853, 0.54591846, -0.70341337, -0.44107112, -0.42153281, 0.50929374, -0.30746633, -0.0030613283, 0.15443291],[-0.12617019, -1.5108511, 0.33994848, -0.38738501, -0.53060532, 0.32151502, 0.67816746, 0.21156509, -0.79692817, -0.40713632, 0.2992413, -0.37576288, -1.4579811, 0.074829251, -0.43288171, -0.059913628],[0.67710286, 0.35013556, 1.1249863, -0.4300749, -0.50567698, 0.17638367, -0.038496081, 1.006829, 0.91364032, 0.61151242, -0.86668867, -1.7892594, 0.23001204, -0.14297058, 1.1624572, 0.37031972],[-0.6875385, -0.88249487, -0.2549369, -0.82489288, -1.4000617, -0.55393744, 0.30009788, -0.22803657, 0.16315658, -0.43697569, -0.20840386, -0.087542109, -0.82705814, -0.80379045, -0.13612179, -0.47483432],[1.2657987, 0.097149089, -0.35491669, 0.13615538, 0.708363, 0.192185, -0.84233457, -1.5393209, -1.3990552, 0.14441337, 0.69704747, 0.2189088, 0.74482656, 0.60463905, 0.43360138, 0.002574391],[0.65412581, 0.0011200159, 0.85179967, -0.052585512, 0.24365586, 0.74750477, 0.64062214, -0.41642079, 0.13102216, -0.93886429, 0.38455853, -0.0024161532, -0.49778241, -0.43158406, 0.33588123, -1.5947367],[-0.32651901, -1.1093949, -0.63783163, -0.87559927, -0.40075836, -0.32517555, 0.38549, -0.23857348, 0.577196, -0.82855707, -0.43482891, 0.13146892, -0.09457799, -0.81895089, 0.53240299, -0.60139191]]
	b_2=[-1.88859,-1.56998,0.344342,-1.41068,-3.24009,-1.08195,0.667781,0.688689,0.31313,1.13941,-0.424968,0.299277,-1.49996,0.672552,-0.753661,-0.45994]
	W_3=[[0.43510112, -0.12689726, 0.23136407, 0.85634816, -0.4640618, 0.36753252, -0.68611133, 1.0627078],[-0.89668632, 1.4336659, 1.023508, 0.46596766, -0.72395039, 0.36314505, 0.096933261, -1.1769335],[-0.27558094, 0.17050451, 0.36278197, -0.41876087, 0.1835039, -1.4776642, -0.82238615, -0.52107215],[-1.2005215, -0.051464964, 0.76481718, -0.94654715, -0.38368744, 0.53199488, 0.80345911, -0.56817728],[-0.51357299, -1.5010945, 1.3564659, 0.069663331, 0.5832845, -0.42021325, 0.72589451, 0.48251963],[-0.28277922, -0.15226732, 0.78058052, 0.60646945, 0.75404483, -0.79090428, -0.79350239, -0.24120696],[-0.19004333, 0.042720657, -0.65874511, 0.27635279, -1.0038425, -1.5116607, 0.68428385, 0.4369095],[0.015159793, 0.029818365, -0.64918226, -0.39169604, 0.52310109, -2.450588, 0.57363355, 0.37732378],[-0.76501113, 0.49504018, 0.38968325, -0.46347654, -0.76691252, -0.9726308, -1.3481565, -0.84109598],[-1.1956623, 0.64316261, -0.29512712, -0.80199963, -1.3055751, 0.14256962, -0.028309094, 0.29254097],[0.66000861, -0.11587395, -1.0413709, 0.35267824, -0.11837911, 0.20326887, -0.25653064, -0.30227968],[1.0452631, -0.23373069, -0.920766, -0.31990653, -0.15749966, 0.00023125597, -0.18165421, 0.58889705],[-0.94356292, -0.42784753, 0.43211406, -0.024808569, -2.0665803, 0.15157433, 0.37239102, -0.095083162],[-1.8263603, 0.055414058, -0.56409585, 0.42948005, 0.066796519, 0.10846011, -0.20341024, 0.31350732],[-0.18223985, 0.18647708, 0.49950984, -1.9507619, -0.60921043, -0.60834283, 0.10945106, 0.15308574],[-0.5233742, -0.12785909, -1.0117637, 0.77302456, 0.4201971, -0.37544242, 0.20182239, 0.4540315]]
	b_3=[0.513653,0.74531,-0.626461,-0.214508,-1.12127,1.58705,-0.492243,-0.0527776]
	W_4=[-0.10890146, 0.26151422,-0.33368167, 0.2530179, 0.2823076, 0.18451536, 0.50400466, 0.28650227]
	b_4= 0.05139918
	a_1=[0.0]*32
	for i in range(32):
		sum=0.0
		for j in range(32):
			# print(x[j])
			sum+=x[j]*W_1[j][i]
		for j in range(32):
			sum+=y[j]*W_1[j+32][i]
		sum+=b_1[i]
		a_1[i]=Relu.Relu(sum)
		# print(a_1[i])
	a_2=[0.0]*16
	for i in range(16):
		sum=0.0
		for j in range(32):
			sum+=a_1[j] * W_2[j][i]
		sum+=b_2[i]
		a_2[i]=Relu.Relu(sum)
	a_3=[0.0]*8
	for i in range(8):
		sum=0.0
		for j in range(16):
			sum+=a_2[j] * W_3[j][i]
		sum+=b_3[i]
		a_3[i]=Relu.Relu(sum)
	score=0.0
	for i in range(8):
		score+=a_3[i]*W_4[i]
	score+=b_4
	return Sigmoid.Sigmoid(score)
