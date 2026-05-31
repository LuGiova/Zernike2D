# Pull Request: Ottimizzazione caching Zernike e verifica risultati

Breve descrizione
- Applico un'ottimizzazione che riutilizza l'oggetto di cache interno di `zepyros` (`zernike_obj`) all'interno di `get_all_invariants`, evitando di ricostruire ripetutamente le strutture e i polinomi pesanti.

Modifiche principali
- File modificati:
  - `get_binding_propensity.py`: `get_all_invariants` ora mantiene e passa `zernike_obj` tra le iterazioni.
  - `requirements.txt`: puntato alla versione ottimizzata di `zepyros` per installazioni future.

Motivazione
- Profiling (cProfile/pstats) ha evidenziato che il costo dominante era all'interno delle routine di `zepyros` (`get_zernike`, `compute_coeff_nm`, ecc.). Riutilizzare l'oggetto di cache interno riduce fortemente le computazioni ripetute.

Risultati e benchmark
- Confronto numerico: i file risultanti sono numericamente equivalenti agli originali (`np.allclose(..., equal_nan=True) == True`).
  - `1a1u_A_bp.csv`: max abs diff ≈ 2.056e-08
  - `1a1u_C_bp.csv`: max abs diff ≈ 2.117e-08
- Tempi di esecuzione:
  - Mean originals: 7874.275 s
  - Mean ottimizzato (output_files): 102.850 s
  - Miglioramento: ~98.69%

Test eseguiti
- Eseguiti script di confronto CSV e confronto `execution_times.csv` (vedi esempio di comando usato in workspace).
- Eseguito profiling esplorativo con `cProfile` per identificare hotspot e verificare l'efficacia della cache.

Come provare localmente
1. Crea e attiva un ambiente virtuale Python. Installa le dipendenze con:
```
pip install -r requirements.txt
```
2. Esegui lo script principale su un input di prova:
```
python get_binding_propensity.py -sf1 ./input_files/1a1u_A.csv -sf2 ./input_files/1a1u_C.csv -o ./output_files/
```
3. Confronta i CSV prodotti con gli originali (es. script di confronto incluso nella sessione):
```
# esempio rapido in Python
import pandas as pd, numpy as np
orig = pd.read_csv('originals/1a1u_A_bp.csv').to_numpy()
out = pd.read_csv('output_files/1a1u_A_bp.csv').to_numpy()
print(np.allclose(orig, out, equal_nan=True), np.max(np.abs(orig-out)))
```

Punti di attenzione e rischi
- La modifica opera sul flusso interno a `zepyros` tramite il parametro `zernike_obj`. Questo presuppone stabilità dell'API di `zepyros`; se l'API cambia, potrebbe essere necessario adattare il codice.
- Non sono state introdotte modifiche algoritmiche ai coefficienti prodotti, solo riuso di cache.

Suggerimenti per la review
- Verificare che i CSV prodotti siano allclose rispetto agli originali.
- Validare i miglioramenti temporali su macchine di riferimento.

Autore: modifica eseguita da Giovanni Marzioni
