# WMT Shared Task: Similar Language Translation
### For Romance Languages: Spanish → Romanian

**Generating a competition pipeline for automated training on multiple GPUs with logging systems.**

A more detailed description of the project and the obtained results can be found in the documentation paper.

### Models currently implemented
- [x] Transformer models with Multi-Headed Attention (implemented from scratch)
- [x] Recurrent Neural Networks with Attention

### Other frameworks used for comparison
- JoeyMT
- HuggingFace
- OpenNMT

### Contribuitors
- [Adrian Iordache](https://github.com/AdrianIordache)
  1. Designing framework architecture
  2. Writing the code for distributed training on multiple GPUs with automated logging system
  3. Writing the code for inference/translation using k-beam search
  4. Researching and writing code for the Transformer architecture with multi-headed attention layers
  5. Training ideas for relevant experiments with Transformers to the proposed problem
  6. Establishing some points to define the cleaning of the data collection
- [Andrei Gîdea](https://github.com/andreiG98)
  1. Researching and writing code for the RNN architecture with attention layers 
  2. Training ideas for relevant experiments with RNN to the proposed problem
  3. Training experiments with NMT Joey and Hugging Face frameworks for comparisons our implementation
  4. Writing the documentation and the presentation
  5. Establishing some points to define the cleaning of the data collection
  
### Translated Examples
| Category           | Text                                                                                                                                                                                                                                                                                                                                          |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Source                    | El sector agrícola tiene importantes efectos directos en la biodiversidad y los ecosistemas.                                                                                                                                                                                                                                                               |
| Reference                 | Sectorul agricol are un impact direct și semnificativ asupra biodiversității și a ecosistemelor.                                                                                                                                                                                                                                                           |
| Output System 1           | sectorul agricol are efecte directe semnificative asupra biodiversității și ecosistemelor.                                                                                                                                                                                                                                                                 |
| Output System 2           | sectorul agricol are efecte directe semnificative asupra biodiversității și ecosistemelor.                                                                                                                                                                                                                                                                 |
| Output System 3 (JoeyNMT) | Sectorul agricol are efecte directe asupra biodiversității și ecosistemelor.        

| Category           | Text   |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Source                    | A fin de garantizar la aplicación uniforme del presente artículo, la AEVM podrá elaborar proyectos de normas técnicas de regulación para especificar con más detalle la información que se deberá facilitar a las autoridades competentes durante la solicitud de registro tal y como se establece en el apartado 1 y para especificar con más detalle las condiciones que se establecen en el apartado 2.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Reference                 | Pentru a asigura aplicarea uniformă a prezentului articol, AEVMP poate elabora proiecte de standarde tehnice de reglementare pentru a preciza și mai mult informațiile ce trebuie furnizate autorităților competente în cererea de înregistrare astfel cum este prevăzută la alineatul (1) și pentru a preciza condițiile astfel cum sunt prevăzute la alineatul (2).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Output System 1           | pentru a asigura aplicarea uniformă a prezentului articol, comitetul de standardizare poate elabora proiecte de norme tehnice de reglementare pentru a preciza în detaliu informaţiile care trebuie furnizate autorităţilor competente în timpul cererii de înregistrare prevăzute în alin. (1) şi pentru a preciza în detaliu condiţiile prevăzute în alin. (2).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Output System 2           | în scopul asigurării aplicării uniforme a prezentului articol, $\mu$ poate elabora proiecte de standarde tehnice de reglementare pentru a preciza în detaliu informaţiile care trebuie furnizate autorităţilor competente în timpul cererii de înregistrare prevăzute la alin. (1) şi pentru a preciza mai detaliat condiţiile prevăzute la alin.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Output System 3 (JoeyNMT) | Pentru a asigura aplicarea uniformă a prezentului articol, AEVMP poate elabora proiecte de norme tehnice de reglementare pentru a specifica mai detaliat informaţiile care trebuie furnizate autorităţilor competente în timpul cererii de înregistrare în conformitate cu alin. (1) şi pentru a specifica mai detaliat condiţiile stabilite în                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |


| Category  | Text                                                                                                                                                             
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Source           | Actualmente, se dispone de nuevas herramientas que pueden facilitar la presentación de capacidades y cualificaciones utilizando distintos formatos, digitales y en línea. |
| Reference        | În prezent, sunt disponibile noi instrumente care pot facilita prezentarea aptitudinilor și a calificărilor folosind diverse formate online și digitale.                  |
| Our System       | În prezent, există noi instrumente care pot facilita prezentarea capacităţilor şi calificărilor prin diferite formate, digitale şi online.                                |
| Google Translate | Sunt disponibile acum instrumente noi care pot facilita prezentarea competențelor și calificărilor folosind diferite formate, digitale și online.                         |
