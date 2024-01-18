/*
100 observations of 15 dimensional features

weights = [5, -3, 5.8, -2.9, 7.1, 3.8, -8, 1.2 ...]
*/

CREATE TABLE my_table (features DOUBLE[], label DOUBLE);
INSERT INTO my_table VALUES ([-2.411869365610726, 2.092534438888792, -0.13017750832049813, 1.2742261055193234, 1.156023233066231, 2.3659194130313077, 1.0951989131106985, 0.17179808064481367, 1.6989773005791007, -2.2643566336644008], -21.45398769715015);
INSERT INTO my_table VALUES ([-0.6237303749125681, -0.12204739265303248, 0.6724715136597232, 1.1474934761645215, 0.5866392899561507, -0.08914913621709315, -1.126638383294642, -1.101528711865675, 0.03661372741369009, -0.7234629180799625], 12.013807209467892);
INSERT INTO my_table VALUES ([-0.8920038975453985, -1.2639452351364442, -0.261705738350126, 0.25506872590740653, -0.14076935195392587, 1.1447576171739022, 1.0769459328547115, 0.4168800187403821, -0.75183574962964, 0.7893780363739051], -3.5626850952781153);
INSERT INTO my_table VALUES ([1.9252707599572025, -0.12162546044793891, -0.12194595187341606, 0.21712819854082246, 1.0376917578778786, -0.3508567612333919, 0.06503392811296849, -0.7426428567513149, -0.5435539663357625, 0.7156343852847823], 15.652704668264318);
INSERT INTO my_table VALUES ([-1.212581255350511, -1.372722864652732, 1.1857130224655272, 0.5793032436905671, 0.00016666229805088956, 2.5466231829079193, 0.42843582732762464, 1.4666127264888766, 1.1418831129241365, -0.37889106867560807], 1.5499181618235673);
INSERT INTO my_table VALUES ([0.44512178146629333, 0.4657325637363819, 0.5937719970099286, 0.24715854054444322, -1.1070087902230603, 0.9795641013349369, 1.2815973200010333, -0.5824803627479155, -0.20937613681445444, -0.3120805237222054], -8.15008283551935);
INSERT INTO my_table VALUES ([0.25180193985750454, 1.4310052903567556, 0.9875901977689718, 0.1905004980960992, 0.3607848755470529, 0.8194725732377014, 1.0925024169494424, -1.3203787601397026, -0.04074592201414747, 0.021078312491669325], -2.1924842068021895);
INSERT INTO my_table VALUES ([1.2161194144165042, 1.1600095757280942, 0.1937763912593146, 0.1970341235018472, 0.8303139936734547, -0.6531686423883211, 0.43593424511578005, -0.2079536285437718, -0.9618009517124542, 2.8570501451385177], 0.35146146724556715);
INSERT INTO my_table VALUES ([-0.15945900367280877, 0.5226153964600877, -0.08951805820610388, 0.18897060350028164, -2.0012507216789936, 0.49860402030926115, -0.1658993922505377, -0.2908673411027094, -0.4409161232563117, 0.5089587341004492], -12.54094827653954);
INSERT INTO my_table VALUES ([2.0001152956247505, -0.7484589046872983, 0.46581741515946207, 0.7351415320435544, -0.5876674973911489, 0.16211348543343185, -0.24252350715934504, 1.0320131030323376, -1.8549394235788381, -1.1210905373761966], 35.5104601016782);
INSERT INTO my_table VALUES ([-1.1114319089312343, 1.1653793135637012, -1.3584064385917975, -0.38273682441878265, 0.40997847378827135, 0.7223360392108248, -1.1905877013293968, -2.3015211259637693, 0.24326927015303393, -1.0443953801813393], -1.4254228437421306);
INSERT INTO my_table VALUES ([1.2534624303422972, 0.18398659738483653, -0.2863872756439118, -0.34892196024826044, -0.09741389741413338, 0.812255806250274, 1.9459153230625035, 1.7409173647830058, 0.6323962968661153, -1.1507841700497599], -7.444538333873034);
INSERT INTO my_table VALUES ([-0.46441492657561634, -1.00189395703819, 1.1016861821732533, 0.7992949540030402, 0.76956165242727, -0.19269905648852512, 0.07416441689716088, 0.845345897957694, -0.0329639135185115, -1.7191537569137714], 17.45495131105281);
INSERT INTO my_table VALUES ([0.41009498402178385, 1.4030586361526338, -1.1940064181583105, -0.7596776654524701, 0.3289250232314785, 1.3997212726008237, 0.3381285837095446, 2.6088711617814084, -0.685960307843625, -0.8939312518583239], 11.74457054688979);
INSERT INTO my_table VALUES ([0.2087979369320243, 0.24815437098556697, 1.33665341562261, 0.58811413269272, 0.8726874992194484, -1.1928901449760128, 0.8688711564841002, -0.2066600974499803, -1.690995499987166, 0.2731125707292707], 16.404505375052135);
INSERT INTO my_table VALUES ([-0.5477106017092143, 1.0196627362953625, -1.1179237860184759, -1.265173340624549, -0.7256714678219544, 0.9778622585259508, -1.4299292488399853, 0.05607028945483394, 1.2907703582632692, -0.014886828435482672], -11.258270856977045);
INSERT INTO my_table VALUES ([0.08820932038058284, -0.6127994873113962, -1.4387725212352194, 0.928265601601614, -0.8950990067257456, -1.2316698946134303, -0.9277408855326171, -0.9418266677189909, 1.6418433020657774, 0.7919151496145641], -33.0815115870899);
INSERT INTO my_table VALUES ([0.5582511903999522, -1.1775797445794485, -0.024806232480858854, -0.26301034962491165, 1.0595052400252647, 0.10956810656162573, 0.017494717590565412, -0.6987541001705159, 1.381650827855028, 0.6150264775164048], -2.3582220018878948);
INSERT INTO my_table VALUES ([1.9197033663947596, 1.333124673559124, -1.813719388226879, -0.7277038405736649, 2.6947542689886275, 0.6464675629251891, -1.3731212635492924, 0.7155489832124153, 0.7544343389152659, 0.4528681910763039], 21.251926076646757);
INSERT INTO my_table VALUES ([-0.39183750913859505, 0.29622775479441715, -0.9127859999747582, -0.39620537185211946, 0.24009363010457924, 0.6586092705175518, -0.9296284205193931, 0.04479479377746812, -0.42518022283433776, 0.06568781207007088], 8.638522482204284);
INSERT INTO my_table VALUES ([-0.718343903035138, -1.6123900861212876, -1.2854633180954964, 1.2253622835835167, 2.174336328093889, -0.6548975293207465, -0.25075016349062496, -0.48399276875249786, -0.787722155860443, 0.41073051721909687], 10.683981350010054);
INSERT INTO my_table VALUES ([2.193713684206495, 1.225544041361808, -1.914314567000131, -0.707220630853677, -0.5085181968739076, -0.4368479798216568, 1.519334272135338, 0.42174567954514103, -0.6709624665537298, 1.4241102883746277], -18.01796403722513);
INSERT INTO my_table VALUES ([-0.8474942888572264, 0.9305029843303139, -0.15615358033394958, 0.9224565410219678, -0.5402856977599642, -1.0043402647538482, -0.16281023859067031, -0.5710709692406175, -0.8035627739218044, 1.6104538832322863], -16.45375469445117);
INSERT INTO my_table VALUES ([-0.4927496780055312, 0.42677186051068106, 1.8150711831426418, -0.497894319328405, 0.6058371486624136, 0.13931564599999413, -0.9812118852201048, 1.103969336025247, 0.6323057731944352, 0.5813047257142638], 13.531236909389431);
INSERT INTO my_table VALUES ([1.7981575117198703, -0.6112887801847013, -1.7341226432317958, -0.9539808614492314, -0.1749090791036992, 0.7281245280201648, 1.0253520660636437, 1.6262037407100713, 0.5631399606762552, -0.28633763822653796], -5.565533758348359);
INSERT INTO my_table VALUES ([1.343596058677312, -1.8905691004134575, -1.258684007759465, 0.26060183909803586, -0.3553257421677014, 0.057087425547746415, -0.9223577519256216, 0.2666358122617368, -0.049236652764006995, -0.1422977314184735], 10.811612788368803);
INSERT INTO my_table VALUES ([0.22706521270835328, -0.936970358740652, 1.0217292845203947, 0.08107750645469305, -0.46087226172663565, -0.49561566685250197, 0.796877815393543, -0.1709351552577217, -1.2659650102167381, 0.33305922846037367], 9.035669761119312);
INSERT INTO my_table VALUES ([-2.233482732759647, 1.735563392254685, -0.46967961637794325, 0.4870716113322201, 1.1767682667159407, -1.3560391085081271, 0.05085285855341329, 0.07203289803344129, 0.04543328360994356, -0.32277340370843993], -16.72317181001567);
INSERT INTO my_table VALUES ([-0.7764118136803071, 0.12022788577742885, -0.13125876859277708, 0.46025893038264737, 0.7098191696918826, 0.7987282278550351, 0.2656908741233258, 0.30061088684917864, -0.919078085630752, 0.8286723752770359], 5.589744031793196);
INSERT INTO my_table VALUES ([1.892793388182338, 1.006453823438036, 1.0666661625457423, -1.2748625277662304, 1.2715375085677516, 1.6921286019371324, 1.8518973355090667, 1.2856385848854424, -0.36651335352607484, -0.41008130730119347], 23.864786850440254);
INSERT INTO my_table VALUES ([0.5926316289755931, -1.210160586765608, 0.8335481153133847, -0.18328772841271065, -1.2939598043357257, 0.4599636118770198, 0.18734269169395862, 0.3628322237737353, 0.5469811776560808, 0.5685801136458652], -4.345992500392715);
INSERT INTO my_table VALUES ([0.5018418756046015, -0.929324774653521, 0.36952685352301157, -1.2700235430071722, 0.9858376203682826, -0.08865020062717845, 0.8278771481551561, 1.8618855827627747, -0.7500508205131569, -0.6449778336545926], 23.531739607872993);
INSERT INTO my_table VALUES ([-1.762096589528697, 0.23894094088137852, -0.4780712288531984, -0.6708523225592844, -0.969889968576587, -1.608428769126073, 0.8973632436052643, -0.32645158041546746, 1.496405590781331, -0.8255968487389748], -42.27045169115925);
INSERT INTO my_table VALUES ([0.831286000893126, 0.37086323288128975, -1.2169248189371702, 0.1335893508771618, 0.9064303342273744, -0.8662431614155053, 0.09359812501706648, -0.09544516023450356, -2.2771087517620594, 1.379495143833347], 14.628376444004237);
INSERT INTO my_table VALUES ([-0.29417293899720415, 0.35914904518880364, -0.4851469534639548, -0.5857027635027472, 1.2322309496208428, 0.824424510286682, 2.241246381386544, -0.38839518057733247, 0.379610365784775, 0.7068334262934621], -16.90486154457499);
INSERT INTO my_table VALUES ([0.7685693589262987, 0.5644429564408859, 0.16891807844212525, 0.21085979880807762, -1.4514849050632852, 0.129974899363409, -2.4374751562991643, 0.48443279022724667, 0.3301054595779366, 0.24430291041600585], 8.493115480458583);
INSERT INTO my_table VALUES ([0.8545280371911625, -1.0103127299466013, -0.366060356805351, -0.6129614397163735, 0.16210378745985768, -1.1671942392211025, -0.8523014624525369, -1.4215057219961984, -1.1990352197559444, -0.011043421801675194], 20.703051143802604);
INSERT INTO my_table VALUES ([-0.8959519948899731, 1.86836278857616, 0.9663205722088306, 0.8433865135873346, -0.9058361749753637, -1.1219814499510303, -0.7785588750063046, -0.2008105225735221, -0.7944331070070774, 0.7999680591605427], -7.128455979051397);
INSERT INTO my_table VALUES ([0.4665532632250862, 1.5765692617766247, -0.8181992045586977, -0.938375257737274, -0.6224079072007299, -1.193122805195, -0.3071228675122325, -0.45567662466007164, 1.1004527283062784, -0.32250913536294573], -21.00394406027499);
INSERT INTO my_table VALUES ([-0.16103504509607397, -0.29131759180008815, 0.08829855437183837, -0.014571882793772679, -0.6719392426732829, -0.13742427144127553, 2.5556263531719643, 1.6081952040106284, -1.1183746909365955, 0.9008703836869779], -15.89673598085463);
INSERT INTO my_table VALUES ([0.17033151933712803, 0.18211521463699593, 0.13182232773444896, -0.4152310194882298, -0.11695359381401618, -0.4231153207704939, -1.4153712504844462, -2.2588952253017034, 0.04804349060179422, 0.9030644689233605], 4.179637082220582);
INSERT INTO my_table VALUES ([1.2031796909768575, 0.3329260197487465, -0.8056164031171508, -0.0012224583539408307, -0.6329158562469303, 0.6468817529874786, 1.5810673519023515, -0.13456583091666946, -1.376090448715515, -1.364079874138992], 4.854951570132767);
INSERT INTO my_table VALUES ([0.018131139554363487, 0.28122871294380675, 0.8343988813821434, 1.1697646526191192, -0.7586341542852691, 0.08288109145235502, -2.352146823902545, -1.9481682815878536, 0.0658578767217119, -0.12077898565016561], 11.957462643955605);
INSERT INTO my_table VALUES ([0.6113832518071085, -0.11779725253278747, 1.4669361328726087, 0.9241740697092989, -0.9522679961397122, -0.4866012894751271, 1.171490446174314, 0.5224300996876898, -2.5324173433217365, -0.5718903599024673], 19.356108873246903);
INSERT INTO my_table VALUES ([0.2012130331475622, -0.8011642674610064, 0.047205940810992275, -0.5154624945105302, 0.644519613876799, 0.820737665557791, -0.9259193481477532, 0.7501891528669576, 0.34000698587605516, 0.6390713732083164], 15.130498887524503);
INSERT INTO my_table VALUES ([-0.2953172511049426, 0.25504770292828555, -0.3779143544628606, -1.2301343529382587, -0.08385245366629544, -0.027735405545660195, 1.0789559255022643, -0.3956607528012934, 0.40712222451443875, 0.07944061868231718], -15.037590887315758);
INSERT INTO my_table VALUES ([-0.7400274847399124, 1.578976048203826, 0.34259544638751455, -0.5549615514574134, 0.24653971276578324, 1.0241962697288582, -1.5311146534979965, 0.6145022961231699, -1.5338272069287036, -0.8219470015963336], 32.4251410442003);
INSERT INTO my_table VALUES ([-0.3579084463581107, 1.12273930442918, 2.2959599428979662, -0.16680246222392656, 1.7217097611423193, 0.291860321607951, -0.02316991532353169, 0.3173428120107937, -1.1940267139755572, 0.5028339567412543], 32.25087573744614);
INSERT INTO my_table VALUES ([-0.6395896683372055, -0.6592332590821884, -2.1872967221471997, -0.24679689049163828, -1.5134874853122562, -0.07222390921293988, 0.015310365837938489, 0.8288983255358031, 0.6717460763215407, -0.3245107747392443], -28.62621640893263);
INSERT INTO my_table VALUES ([-1.85412504175886, 1.6794305899914634, -1.1030394560231296, 1.1155211407459398, -0.5503161143369925, 0.22532852921158691, 0.31494742338056675, -0.6682885518504362, 1.3386780688867093, -0.6971593369691498], -40.63892244862773);
INSERT INTO my_table VALUES ([-0.513031781878844, 0.14856278865583403, -0.5533899156542482, 0.393897074109945, -1.3212391738498699, 1.2487852024951638, -0.785914444166365, 0.9234854069415968, 1.1266970770327416, -0.16437835716794], -15.066638586785615);
INSERT INTO my_table VALUES ([-1.026353695437636, -0.6388606946781666, 0.2890408153556597, -0.1037082257941722, -0.27688876037308047, -1.007995730072428, -0.08003670209853676, -0.050847439261585095, 0.4703516965146332, -0.43937879226868054], -9.266103960503871);
INSERT INTO my_table VALUES ([1.275876317370106, 1.4305166918968937, 1.619488455892341, -1.2235313233378413, 0.8479572131490748, -0.027982359849137422, -0.8162688179460055, 0.8444364620095891, 1.2172363259102426, 0.6478268883383166], 13.715230379394091);
INSERT INTO my_table VALUES ([-0.2528906220871912, 0.47650295760843125, -1.2468387343072387, -1.9788440953184476, -0.48179156685469743, 1.3049906589062124, -0.11878148045501502, -1.537581467650294, 0.5553690172414242, -0.40555373983215826], -7.338407867579749);
INSERT INTO my_table VALUES ([0.8065817702329232, -0.13495918294075143, 0.12340281024138255, -0.5248522285878541, 1.5368870590590478, -0.6588833809258146, 0.6131120534228203, -0.04783340093369246, 0.8913391591101717, -0.5855552502723097], 3.7565133015428174);
INSERT INTO my_table VALUES ([-0.06301713843629492, 1.41408291053814, 0.31502559001526964, 0.7414969371159342, 1.7006185811378876, 0.40511109153286806, -0.20392983141102017, 0.28343814269967627, 0.17340981093732483, 0.2096165201792197], 8.107704864055892);
INSERT INTO my_table VALUES ([-0.4323715266831944, 1.7324482255585576, 1.274697534707094, 0.11266561069649744, -0.4239510789956238, 1.2721646237884208, -1.019575361994184, 0.8006682513364898, 0.21949942821753682, 0.9242030520542912], 4.59419366929235);
INSERT INTO my_table VALUES ([0.1671027294684581, 0.4963116888266063, -1.4412503392731888, -0.8708857339699088, 1.4451653252554786, -0.7667531591128192, -1.578749698025716, 0.41070954929955156, 0.6467799165369773, 1.499212654793859], 1.282941765790806);
INSERT INTO my_table VALUES ([-0.9573032981328228, 0.23792718699586457, 0.3805824495144887, -0.1555191576383295, 0.1444255410234441, -2.09463421251922, -0.8246290740088422, 0.26638624447553566, -0.616217680250201, 1.514434184475044], -3.119475408499915);
INSERT INTO my_table VALUES ([-1.416649546574819, 0.14313933787612884, -1.8689782458779114, -1.1748132083808815, -1.9280535788390543, 0.9715495815896699, 1.0842503473590548, 0.8855247679763503, 1.6723612291176777, 0.8258133825853955], -52.57923870043084);
INSERT INTO my_table VALUES ([0.08520741301702245, -0.7689118363934209, -0.722374136522662, -0.5377131882169128, -1.6178277667579508, 0.039199731591320765, -0.8573822784830357, 1.2261408555440663, -0.4198766104054227, 1.4644171793240204], -4.898593791301486);
INSERT INTO my_table VALUES ([-1.3449846064126905, -0.842207019280081, 0.8976694880590825, 0.10434103867679367, -1.0328897653680225, -0.8518008820836858, 1.1541780292788324, -0.979957877721286, -1.6731969735715189, 0.5761598583032599], -6.1293636956860515);
INSERT INTO my_table VALUES ([1.867069808393755, 1.2076300439671324, -1.1841338543054745, 0.6859386443315109, -0.8408600391762453, 0.11236504785019852, -1.474125410790453, -0.9073710174180277, -0.17159484064036074, -1.0362666122761266], 8.067408149534948);
INSERT INTO my_table VALUES ([-0.5446614166630755, -1.755084822681202, 0.5411665979706795, -0.9944070029265538, -1.3472912823447285, 1.0012357351790302, -0.3071346081271016, -0.012921614640188213, 1.7040890127204318, -0.9824444455743119], -7.4992219765936605);
INSERT INTO my_table VALUES ([-0.33283781515944005, -0.4062456311101702, -0.20341898490854182, 1.1130360952313987, -1.1404574777925072, -1.1915239596717473, 0.146930068239779, -1.5525314431378787, 1.2709805882197893, -0.8590786016285531], -29.491182084406404);
INSERT INTO my_table VALUES ([-1.1315864020801452, 0.3479849282413161, -0.8747481744250772, 1.7494302372684363, -0.1684913116048206, 1.4213404328726962, 1.303131181192977, 0.7856630946424004, -0.4175715196318367, 0.1941230232279343], -18.807580951521647);
INSERT INTO my_table VALUES ([0.34872710496394327, -1.438515146596603, 0.6898471863644279, 0.707980534249027, 0.9054984810775928, 0.6584357812692354, -0.7400521189294595, 0.2614142436054012, -1.948594947513979, 0.791673339624943], 39.138422278400824);
INSERT INTO my_table VALUES ([0.8000242230038782, -3.0687179656562886, 0.9472768296742007, -0.2740775911010166, -2.2373433028426675, 0.10076283778840726, 1.9929048613173603, 0.8238535418137201, -0.5746675237474937, -0.37246139628917696], -3.708001319143108);
INSERT INTO my_table VALUES ([-1.4855027766114013, -0.8399256210447834, 1.0210783100005008, 0.2967684635079639, -0.10831431378297424, 0.8417890847468933, -0.28327746982707414, 0.6838725694860889, -0.2181941202949719, -1.355606985097238], 13.524193498414752);
INSERT INTO my_table VALUES ([-0.7666268074323224, 1.4491510323487686, -1.5923728264805628, -0.049455780433362156, -0.6707134372650402, -0.858441706300792, 1.4445154542517755, 0.055440515531708735, 0.21996793734405495, 0.8890323675317948], -42.698285191779064);
INSERT INTO my_table VALUES ([-0.7087115310775209, -0.18118311304342022, 0.05475156150484219, 0.7015887024782932, 0.05002231899342786, -0.01994590528234479, -0.34363240320154625, -0.11785327663883044, 0.22279815010629864, 0.2084850987570504], -4.911396279122077);
INSERT INTO my_table VALUES ([-0.5080165828651527, 0.5188069616083654, -1.2435217962493983, -1.4703385396949684, -0.984650705608784, -1.1933305206985472, 0.5243209386617457, 0.20424096058774058, -0.10721776127224768, 0.5222165072534916], -23.651956291216894);
INSERT INTO my_table VALUES ([1.2151600262171558, 0.976120813142995, 1.829531920953452, -0.9299467148426824, -0.3367366813494026, -0.0014214403673262284, -0.3284140441889914, -1.4291705446176721, -0.11855227577104416, -0.3507860416049155], 17.6186130000987);
INSERT INTO my_table VALUES ([-0.34908180915894876, 2.323824854032319, 0.8417610656476854, -1.4959280577723477, -1.2554182950836568, -0.1013502584329177, 0.048607453967518865, 0.017021150180243718, 0.230269861764442, 0.9486886468418929], -15.427677136462155);
INSERT INTO my_table VALUES ([0.32039200133221474, 1.0107540587628798, -0.1987064523170512, 0.6556604141897046, 0.5526315207184082, 0.6005376383823142, -0.8460364446089348, -0.416624790736835, -1.6856998522734243, 1.4280424423879525], 18.68050411620166);
INSERT INTO my_table VALUES ([-0.7969613034097226, 2.177649566738004, 2.097424745832822, 1.8743043765057268, -1.6613042961449656, -0.2797213307716138, -0.7134310383902448, 1.486160317960801, 0.7089206667482048, 1.0703271200007762], -20.66922375899582);
INSERT INTO my_table VALUES ([0.1440461949166119, 1.4137066521435, 1.136378339334403, -1.5740917773142629, -0.23960367815278527, -0.5928634767350124, -0.3659694297646957, 0.46138523889483923, 0.4356709031565436, 1.1830309372920116], -2.1195499390592225);
INSERT INTO my_table VALUES ([1.2233546203950783, -0.24860895916990028, -1.2115289662607223, -1.7916271191140625, 0.7742256789548954, 0.2800105917506038, 0.4095109705265027, 1.403118853488061, -0.1893241807008158, 0.07855911439431007], 11.544509158291358);
INSERT INTO my_table VALUES ([0.7915514610965185, 1.2235650700655036, -1.1429252587771168, -0.04383227189900881, 2.453403246722851, -0.5582098136122066, 0.4462768835945374, 0.8591997268825862, 0.42907281920882906, -1.437778386512071], 8.334847859577877);
INSERT INTO my_table VALUES ([-1.2915607248778298, -2.0100796055664643, -0.8587322961053075, -1.2902847759570506, 1.4186042466077882, 0.8700254588585404, 0.3464648812731941, -0.028546761733920668, 0.7743634858793408, -0.6145690322233454], 3.8208168806189775);
INSERT INTO my_table VALUES ([-0.7296811166657534, -0.5665344752440745, 1.8323982018761642, -0.6080206577649847, -0.14077483591384976, 0.19522518686595441, -0.5088844962348658, 0.207416052766546, -0.4548309501966938, -1.1824700997514905], 23.97389775500827);
INSERT INTO my_table VALUES ([0.20048880457320412, -0.7513840650091389, 1.284355783774547, 1.087972023855708, -0.9689794146937915, 0.45884846966188697, -0.16989689349642056, -1.4790813492737414, -0.4565857835034307, 0.007677532250655194], 6.486842385067597);
INSERT INTO my_table VALUES ([-0.8784605896538707, -1.5220515919062436, -1.0455156650103632, 0.5079560731182843, -0.08322595423957567, -0.055978166108782326, 0.6375860571325184, 0.7532213640622544, -1.1080050656086908, 0.40447971327598525], -3.0932204143866544);
INSERT INTO my_table VALUES ([-0.128793378225559, -2.2060698183666507, 0.4870445885277563, 0.5502125648508326, 1.3095077735940734, 1.8451415669792988, 0.5809393389609939, 0.3247794303210957, -0.022801067017131634, 1.3667092840127963], 13.740300061718209);
INSERT INTO my_table VALUES ([-0.127071797891521, -1.0020993601987753, -0.5309238771018825, 0.8562111615168124, -0.30118845432523306, -1.155273564986154, 1.1524327614832028, -0.30549103429357194, 0.6153607295109627, -2.006321359182443], -16.97148218371767);
INSERT INTO my_table VALUES ([-0.5861102241898747, -0.4725632746881298, 0.29788000880758236, -0.47872706016316696, -0.9085642741482791, -2.1120208093025274, 1.201226108058086, -1.5592590030269196, -0.7658603235856268, -2.581907043228405], -5.928227076170188);
INSERT INTO my_table VALUES ([0.5962558370348399, 0.6692113275612813, -0.8663358622833138, 0.38402308281627867, 0.46604559342836743, 2.114001940546478, 0.78747393729074, 0.0689537908859583, 1.092286111639696, -0.6881837221205355], -7.96294647309156);
INSERT INTO my_table VALUES ([-0.39689520949958373, -1.0670307197773918, -0.1282614119840559, 0.5601008901949128, -2.5626538288351144, 0.5299242616957642, -0.46212058263964007, -2.176648621545792, -0.503713232994276, 0.13459878416659524], -11.826290221118118);
INSERT INTO my_table VALUES ([0.8143087843904683, 0.3381762693727232, 0.10919273258176623, 0.15961747938951218, 2.1115809037404625, -0.690919817872147, 0.611878281332983, -0.9913604026609034, -0.8383483102144168, -0.44629471775707097], 19.68359873306743);
INSERT INTO my_table VALUES ([-1.4920658910884594, 2.5237663966789, -0.7368924787415954, -0.7315669039654122, 0.20864174843297603, 1.185851169932117, -0.4259016979428228, -1.6265370518965998, 1.1213785008786132, -0.3677025651689703], -19.29839740478632);
INSERT INTO my_table VALUES ([-0.9711667988788576, 2.0876820714296356, -0.7561454432946569, -1.177880236266543, -0.21204442419160263, 0.9131403134779984, -1.3053639100508871, 0.2673157262310984, -0.28918720511725293, 0.581477976469396], 1.060182666210542);
INSERT INTO my_table VALUES ([-1.4166047481508444, 1.5305166595625348, 0.28614092867978835, 1.225007647987151, 0.7076488571387443, -1.0122390691152707, 0.020572222090953343, -0.041983941384219695, -2.1309194589222304, 0.027970480833799816], 8.373988028224696);
INSERT INTO my_table VALUES ([0.6182593821504062, -0.1727462280425424, 0.8783173752003418, 0.37839241426603565, -0.7985396280386353, -0.5991538233767053, 0.3899611801553134, -2.1085162323078195, -0.7200263597034088, 1.7621944298625174], -6.2628420826063715);
INSERT INTO my_table VALUES ([-2.1791388230916233, -0.711184490874505, 1.1326267888327395, -0.030608306930787876, 0.8003454253124815, 0.6897463601523841, -0.14622028949166335, 0.9173825240953081, 0.086529461967953, -0.22782267488452687], 8.570182416672205);
INSERT INTO my_table VALUES ([0.6549234169574527, 0.27852238720882727, -1.210801424952192, -1.057690762315026, -0.7031280028571509, -1.1924294953358892, -0.07675010932295545, -0.5465789096150985, 0.1380911705571726, -1.5349366776327364], -6.001998392823495);
INSERT INTO my_table VALUES ([-1.2101718065899274, 0.26293355191811496, 0.40447989204876517, -0.03502201562484458, 0.5599172849533767, -0.44167737434287574, -2.586242318613712, -0.3087824433593877, 1.0042192198761164, 0.7234546287432363], 5.2440461305402355);
INSERT INTO my_table VALUES ([0.1215036776659934, 1.199110329391856, 1.2628316114661704, -1.2376487046150648, 0.6580655929613151, -0.5769540761994282, -0.07730123563433478, -0.5625562233014575, -0.2566892586250192, -0.0780419654756066], 13.215974543007553);
INSERT INTO my_table VALUES ([-0.2601532120483081, 1.6963284358884942, 1.339520350471768, -0.3131287774159081, -0.3252542646758518, 1.1663307855901746, 0.19987020495108498, -2.4921805513673134, 0.4301360008919745, 0.018213485415907005], -4.514129523447332);
INSERT INTO my_table VALUES ([1.2225819696733016, 0.027018667096738207, -1.1821589414672242, 1.4048967001701371, -0.8242001074393419, -1.0501991316377741, -0.7974859699339771, 1.013733028916382, 0.5988110275640653, -0.6223708458864825], -10.459350132959418);
INSERT INTO my_table VALUES ([-2.5160910118425037, -1.2650375904404203, -0.20309133166480325, -1.0173779313417113, -1.203027784820564, -0.8330082457992787, 0.6958240603320087, 1.5789233302456431, -0.8928750035235526, -0.9050148671618471], -9.751164126317782);
SELECT linear_regression(features, label, 0.0001, 0.0, 20000) FROM my_table;
