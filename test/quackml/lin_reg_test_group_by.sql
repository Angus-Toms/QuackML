/* 
50 observations of 5 features randomly assigned to 3 classes 

weights = [
    [3, -1, 4, 5, 6],
    [-7, 3, 8, 2, 1],
    [8, -5, 0, 1, 5]   
] 
*/
CREATE TABLE my_table (features DOUBLE[], label DOUBLE, class INTEGER);
INSERT INTO my_table VALUES ([0.9690383242356039, 0.789015464402426, 0.3587503278730467, 0.07869610406404093, 0.3872721862653349], 6.270214457708786, 0);
INSERT INTO my_table VALUES ([0.7600892540086333, 0.5190774558526542, 0.40900067655247496, 0.12511627358934296, 0.9287516155468067], 8.254201104129171, 2);
INSERT INTO my_table VALUES ([0.5806508768433704, 0.1692771189876794, 0.8160150103965725, 0.6189119245195828, 0.6192274020823969], 4.828446553353589, 1);
INSERT INTO my_table VALUES ([0.10261657558256743, 0.7338335669765031, 0.8690808280203719, 0.019728402480237106, 0.03998228629954326], -2.6285953962440227, 2);
INSERT INTO my_table VALUES ([0.35288536845184537, 0.6286191761694974, 0.5410446347092676, 0.7891961755042861, 0.09731024617807338], 7.12405782261298, 0);
INSERT INTO my_table VALUES ([0.25678526368097665, 0.9458424020550003, 0.6627796890821643, 0.8119417929007682, 0.23008097441399833], 7.915826956304418, 0);
INSERT INTO my_table VALUES ([0.6824864125365027, 0.3717145328902135, 0.405602354510197, 0.23569318181058851, 0.2748903956744485], 5.211463796023785, 2);
INSERT INTO my_table VALUES ([0.8050835468639899, 0.09489033410609404, 0.9705031913327388, 0.7733002786761075, 0.5843714410262899], 9.661374188189006, 2);
INSERT INTO my_table VALUES ([0.6362824202364838, 0.8212802617773469, 0.5068941916301203, 0.0043572055685696665, 0.1224496104349635], 3.871627455905215, 0);
INSERT INTO my_table VALUES ([0.8041384085133653, 0.12825858550207136, 0.5315070912721308, 0.9057154378644184, 0.9652098545456222], 1.7845043573641632, 1);
INSERT INTO my_table VALUES ([0.47763864591009675, 0.6600754984471071, 0.19724328526267254, 0.764711090431324, 0.4157460625034314], 3.364173077993719, 2);
INSERT INTO my_table VALUES ([0.39210703040335926, 0.9102686925035944, 0.05635369262320378, 0.7157412028376284, 0.5842498549133549], 7.57567231286757, 0);
INSERT INTO my_table VALUES ([0.8185331869259751, 0.8707891867125382, 0.8522625628943226, 0.14130390861969322, 0.5484220304310018], 4.531765602480759, 1);
INSERT INTO my_table VALUES ([0.006045174067815684, 0.5066875809702851, 0.46735935368492965, 0.3346838866569827, 0.1872902881120061], -1.2139411850918869, 2);
INSERT INTO my_table VALUES ([0.40892310195015213, 0.2493693192685712, 0.7212560150797515, 0.09322381123619838, 0.3837492787393192], 4.036508424191155, 2);
INSERT INTO my_table VALUES ([0.1324985350809632, 0.6000489649348341, 0.1617734384928109, 0.6055258220129641, 0.9197973926875198], 3.2642562414240985, 2);
INSERT INTO my_table VALUES ([0.6491111542963502, 0.24309140865241496, 0.32746397345349365, 0.35362077363999345, 0.19112689493883583], 5.2866874394428995, 2);
INSERT INTO my_table VALUES ([0.16150629318977283, 0.9356289864833559, 0.4114542528463112, 0.6105579999274661, 0.2961218875978995], 6.485214817344979, 1);
INSERT INTO my_table VALUES ([0.022145799247926812, 0.12926676520824887, 0.007455681238084044, 0.5434806937178525, 0.07656412691856374], 1.4559506651482, 1);
INSERT INTO my_table VALUES ([0.3923184625962418, 0.37161335705498577, 0.21987642505852545, 0.7498459774677273, 0.160661765113854], 2.833635718532003, 2);
INSERT INTO my_table VALUES ([0.28434739214333815, 0.4975777778812579, 0.9487086354314233, 0.64962388550542, 0.8646896557258661], 12.586556302156747, 0);
INSERT INTO my_table VALUES ([0.7073975732581098, 0.529053881357013, 0.06845064205391393, 0.5743748603280908, 0.568484086106431], -1.0997824255418056, 1);
INSERT INTO my_table VALUES ([0.3204455648893345, 0.2861878324251509, 0.985740267066608, 0.8147854641701219, 0.9208924943653837], 6.551873292985962, 2);
INSERT INTO my_table VALUES ([0.9658459289132738, 0.17852727649138544, 0.8584779431892632, 0.5976737135462012, 0.8547732058683503], 11.705670791737216, 2);
INSERT INTO my_table VALUES ([0.17705488631783006, 0.21148603369465524, 0.47234540897420196, 0.53854297498071, 0.7699589515643169], 9.521528845445093, 0);
INSERT INTO my_table VALUES ([0.3126105399389857, 0.9129552655354284, 0.24757750287431113, 0.9248180385999071, 0.5526776200800013], 4.93352573730769, 1);
INSERT INTO my_table VALUES ([0.01277175195029645, 0.598384740782292, 0.9741845097471266, 0.5141322237877978, 0.2874407305665696], 7.63197405639551, 0);
INSERT INTO my_table VALUES ([0.15060631522539591, 0.76650033598134, 0.30079407371921796, 0.3920878933848432, 0.8823208149569978], 5.318105992846676, 1);
INSERT INTO my_table VALUES ([0.4854430079080616, 0.7957251732001634, 0.165660501093017, 0.6402814779469611, 0.9174288234580459], 5.132343792500866, 2);
INSERT INTO my_table VALUES ([0.2638316287980206, 0.9147156695522074, 0.24521957279377615, 0.5833965563469868, 0.057518099285451485], 4.119748885464602, 0);
INSERT INTO my_table VALUES ([0.4934692719003352, 0.6458506518132522, 0.9396984767349841, 0.11289542452855506, 0.6402223679100384], 6.866868082984431, 1);
INSERT INTO my_table VALUES ([0.34246794914396983, 0.6883140618968114, 0.9609084263424462, 0.6564581517380162, 0.9502710019096416], 9.61812125780789, 1);
INSERT INTO my_table VALUES ([0.697001962608181, 0.6840713224438479, 0.9272669638489653, 0.3359668120920206, 0.582847722756696], 5.846117286806737, 1);
INSERT INTO my_table VALUES ([0.29659336985465934, 0.6940435281016952, 0.5053902896839282, 0.36916907355060247, 0.8392434464236423], 9.098603786492863, 0);
INSERT INTO my_table VALUES ([0.22989113541846584, 0.23272045274941755, 0.7927859709937495, 0.18099227726673972, 0.8082063256736071], 9.382296177856318, 0);
INSERT INTO my_table VALUES ([0.4059176148961947, 0.5330269430038994, 0.21160223822655366, 0.46162816303620213, 0.41087942149782963], 1.7846111781209988, 1);
INSERT INTO my_table VALUES ([0.7641362206227612, 0.8494999803310478, 0.9507915578202676, 0.8757430265735031, 0.9463494582999926], 7.503714370642954, 1);
INSERT INTO my_table VALUES ([0.777177063677672, 0.49051993250959924, 0.3833586643406811, 0.3676347737126321, 0.7253329385642516], 9.564617415834812, 0);
INSERT INTO my_table VALUES ([0.011681028731031273, 0.487272877697258, 0.6601290420123659, 0.3597028228902768, 0.17368110503305445], 7.55417051888709, 1);
INSERT INTO my_table VALUES ([0.829565902711305, 0.521422880689156, 0.9743264307706804, 0.22672733916291443, 0.6518429730123406], 7.515355022469278, 2);
INSERT INTO my_table VALUES ([0.7937540342583523, 0.8790685798806106, 0.7885734498164612, 0.26365656823782435, 0.16902298997899157], 3.0634608927965483, 2);
INSERT INTO my_table VALUES ([0.6926707181265658, 0.06884703361232936, 0.5667874020616429, 0.11438195279944174, 0.8611018088490909], 9.617021573995775, 2);
INSERT INTO my_table VALUES ([0.5438011101039403, 0.04320845339094859, 0.5140849849017782, 0.14371516871555934, 0.35205698830798415], 1.0751847943985924, 1);
INSERT INTO my_table VALUES ([0.42432131835950937, 0.7480641919615334, 0.6487667149075677, 0.41340404282848864, 0.7470923999096275], 3.8031156294450343, 2);
INSERT INTO my_table VALUES ([0.4509173703783256, 0.4205538869355884, 0.09744272029017664, 0.38178374522909897, 0.7575181571397508], 7.775996774344095, 0);
INSERT INTO my_table VALUES ([0.3086691523616637, 0.6171119589950108, 0.49764153614961715, 0.4080607437600393, 0.9521823994001349], 5.440087986570537, 1);
INSERT INTO my_table VALUES ([0.7103644904645326, 0.7992570209803441, 0.4987855001944891, 0.8121783962805972, 0.8607158036028685], 12.552165254211406, 0);
INSERT INTO my_table VALUES ([0.45071399974440607, 0.3840359500874926, 0.9981909600012187, 0.46505800959199717, 0.448751874159494], 9.97867118206755, 0);
INSERT INTO my_table VALUES ([0.4835688223907536, 0.8123879883840319, 0.6419407347279362, 0.23522768122045112, 0.41370129288766755], 2.1103447828646584, 2);
INSERT INTO my_table VALUES ([0.05804927795785242, 0.18891490829753677, 0.9882796865000606, 0.4922205452421209, 0.7572577483090115], 10.943000887640936, 0);
SELECT class, linear_regression(features, label, 0.05, 0.0, 2000) as linear_regression FROM my_table GROUP BY class;