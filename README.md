# Music Classifier


## Članovi tima

- **Marko Vučić (06/2022)**
- **David Dedić (48/2022)**

## Opis projekta

Cilj projekta je razvoj i poređenje različitih pristupa za **automatsku klasifikaciju kompozitora klasične muzike** na osnovu muzičkih zapisa.

U projektu se razmatraju dva glavna tipa ulaznih podataka:

- **sirovi audio zapisi**,
- **MIDI zapisi**.

Nad oba tipa podataka ispituju se različite grupe modela:

- klasični modeli mašinskog učenja,
- neuronske mreže (potpuno-povezane, konvolutivne i rekurentne neuronske mreže)

Za izbor hiperparametara modela radimo unakrsnom validacijom. A konačnu procjenu modela ugnježđenom unakrsnom validacijom. 

---

## Skup podataka

Primarni skup podataka korišćen u projektu je **MusicNet**.

Skup sadrži klasična muzička djela različitih kompozitora i omogućava rad sa audio zapisima, kao i povezivanje sa simboličkim/MIDI informacijama.

U trenutnoj postavci skupa postoji približno **113 djela**, pri čemu je broj djela po pojedinim kompozitorima veoma neujednačen. Poseban problem predstavljaju manje klase, na primjer:

- Brahms — 8 djela,
- Schubert — 9 djela,
- Mozart — 11 djela.

Ovakva neuravnoteženost otežava treniranje i evaluaciju modela, posebno neuronskih mreža. Brahms se, na primjer, često miješa sa Beethovenom, što može biti posljedica i stilskih sličnosti.

U projektu takođe imamo svesku gdje smo razmatrali rezultate modela kad izbacimo Brahms-a.

Kao potencijalno proširenje razmatra se i bogatiji skup podataka sa većim brojem kompozicija i kompozitora, kako bi se ispitalo da li veća količina podataka značajno poboljšava performanse neuronskih mreža.

---

## Pristupi i modeli

### 1. Klasični modeli nad audio zapisima

Iz audio signala izdvajaju se ručno definisane osobine, među kojima su:

- MFCC koeficijenti,
- chroma osobine,
- spectral contrast,
- spectral centroid,
- spectral bandwidth,
- spectral rolloff,
- tonnetz,
- delta-MFCC,
- spectral flatness,
- zero-crossing rate,
- tempo i onset osobine,
- harmonic/percussive osobine.

Ukupan broj korišćenih audio osobina je približno **147**.

Nad ovim reprezentacijama ispituju se modeli kao što su:

- Random Forest,
- Support Vector Machine,
- XGBoost.

Pored samih performansi, analizira se i važnost pojedinačnih osobina i grupa osobina, kako bi se utvrdilo da li neke karakteristike unose uglavnom šum i mogu biti uklonjene.

---

### 2. Klasični modeli nad MIDI zapisima

Iz MIDI zapisa izdvajaju se osobine koje opisuju različite muzičke karakteristike:

- gustinu nota,
- statistike visine tona,
- melodijski kontur i intervale,
- inter-onset intervale,
- trajanje nota,
- akorde i teksturu,
- histogram pitch klasa,
- histogram instrumenata,
- odnos nota na dobu,
- registar,
- velocity i dinamiku.

Trenutna MIDI reprezentacija sadrži približno **56 osobina**.

Nad ovim osobinama takođe se ispituju SVM, Random Forest i XGBoost modeli.

---

### 3. Neuronske mreže nad audio podacima

Ispituju se dva glavna pristupa:

1. **mel-spektrogram → CRNN**,  
2. **ručno izdvojene audio osobine → fully connected neuronska mreža**.

Glavni problem kod ovih modela je overfitting zbog malog broja nezavisnih kompozicija.

---

### 4. Neuronske mreže nad MIDI podacima

Pored neuronske mreže nad ručno izdvojenim MIDI osobinama, razmatra se i direktniji pristup u kojem se MIDI predstavlja u obliku **piano-roll** tenzora.

Takva reprezentacija može sadržati:

- aktivaciju note,
- početak i kraj note,
- velocity,
- eventualno dodatne kanale sa drugim muzičkim informacijama.

Cilj ovog pristupa je da se zadrži vremenska i melodijska struktura koju agregirane statističke osobine djelimično gube.

---

## Evaluacija

Za evaluaciju modela koriste se metrike kao što su:

- accuracy,
- macro F1-score,
- confusion matrix,
- rezultati po pojedinačnim klasama.

Posebno je važan **macro F1** u našim eksperimentima, upravo zbog nebalansiranosti klasa.

Zato što vremenski ne traju isto sve kompozicije, a prevashodno zato što traju jako dugo, odlučili smo da podijelimo kompoziciju na segmente. U tom slučaju segmenti iste kompozicije moraju se naći u istom skupu tj. ne može se desiti situacija da je jedan segment jedne kompozicije u trening a drugi u test skupu.

Za finalne rezultate planirano je korišćenje nested cross-validation procedure kako bi izbor hiperparametara bio odvojen od završne procjene modela.


---

## Struktura projekta

Planirana završna organizacija projekta podrazumijeva odvojene notebook-ove za glavne eksperimente:

- raw audio + klasični modeli,
- MIDI + klasični modeli,
- raw audio + neuronske mreže,
- MIDI + neuronske mreže 

Zajednička funkcionalnost se izdvaja u module za:

- pripremu i filtriranje skupa podataka,
- ekstrakciju osobina,
- treniranje i evaluaciju neuronskih mreža,
' treniranje i evaluaciju klasičnih modela
- evaluaciju,
- crtanje grafikona,
- pokretanje eksperimentalnih pipeline-ova.

---

## Ograničenja

Glavno ograničenje projekta je mali broj nezavisnih kompozicija.

Iako se iz jednog djela može izdvojiti veliki broj segmenata, ti segmenti nisu potpuno nezavisni primjeri. Zbog toga je broj različitih djela mnogo važniji od samog broja segmenata.

Ovo naročito utiče na:

- stabilnost rezultata unakrsne validacije,
- manjinske klase,
- neuronske mreže velikog kapaciteta,
- izbor hiperparametara.

Rezultate zbog toga treba tumačiti zajedno sa rezultatima po klasama i varijacijama između foldova, a ne samo kroz jednu ukupnu metriku.

## Literatura

https://dorienherremans.com/sites/default/files/Chapter_HerremansEtAl_preprint.pdf
https://arxiv.org/pdf/2010.14805
