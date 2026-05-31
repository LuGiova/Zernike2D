# Report modifiche e risultati

## Obiettivo
Ottimizzare `get_binding_propensity.py` senza alterare i risultati numerici finali rispetto agli output originali.

## Modifiche applicate durante il lavoro

### 1. Profiling temporaneo
- Il metodo `main()` è stato temporaneamente avvolto con `cProfile` per identificare il collo di bottiglia reale.
- Il report `pstats` ha mostrato che il tempo era quasi interamente dentro `zepyros`, in particolare in:
  - `get_zernike`
  - `zernike_decomposition`
  - `compute_coeff_nm`
  - `compute_moment`
  - `r_nm`

### 2. Cache interna di `zepyros`
- Nel metodo `get_all_invariants` è stato introdotto il riuso di `zernike_obj` tra le iterazioni.
- Questo ha eliminato gran parte del ricalcolo interno dei polinomi di Zernike.

### 3. Campionamento della superficie
- Il campionamento della superficie è stato provato a diverse frequenze per misurare l'impatto sui tempi.
- La versione finale del file mantiene il campionamento a passo 10.
- Durante il profiling era stato provato anche un passo più largo per accelerare la misura, ma la configurazione finale è tornata al passo 10.

### 4. Modifiche provvisorie poi rimosse
- La versione con `cProfile` e `pstats` è stata rimossa dal file finale dopo la fase di analisi.
- La variante con preestrazione NumPy fuori dal ciclo è stata testata, ma non è stata mantenuta nella versione finale dopo il rollback richiesto.

## Risultati di runtime osservati

### Baseline indicata dagli originals
- Prima delle ottimizzazioni, gli originals impiegavano circa **3 secondi per iterazione**.

### Profiling con cache non riutilizzata efficacemente
- Un run profiled mostrava un tempo totale di circa **339.871 s**.
- Il costo dominante era nel calcolo Zernike interno di `zepyros`, non nel ciclo Python esterno e non in `cdist`.

### Profiling con cache `zernike_obj` attiva
- Con il riuso dell'oggetto cache interno, il tempo totale è sceso a circa **13.030 s**.
- Questo conferma che il vero guadagno arriva dal riutilizzo dell'istanza Zernike.

## Confronto numerico con gli originals
È stato confrontato il contenuto dei file in `output_files/` con quelli in `originals/`:

- `1a1u_A_bp.csv`
- `1a1u_C_bp.csv`

Esito:
- Stessa forma e stesse colonne.
- I valori non sono identici byte-per-byte.
- I valori sono però equivalenti numericamente entro tolleranza floating-point.
- `np.allclose(..., equal_nan=True)` è risultato `True` per entrambi i file.
- Differenza assoluta massima osservata:
  - circa `2.056e-08` per `1a1u_A_bp.csv`
  - circa `2.117e-08` per `1a1u_C_bp.csv`

## Aggiornamento confronto tempi (finale)

- Sono stati confrontati i file `execution_times.csv` presenti in `originals/` e in `output_files/`.
- Media complessiva dei tempi misurata:
  - originals: **7874.275 s** (mean)
  - output_files: **102.850 s** (mean)
  - Miglioramento percentuale complessivo: **98.69%**

- Confronto per coppia di superfici (media):
  - `1a1u_A + 1a1u_C`: **7874.275 s** -> **102.850 s** (improvement **98.69%**)

Nota: i valori in `originals/execution_times.csv` comprendono run precedenti accumulati; per maggiore accuratezza si può confrontare singole righe o medie per sessione.

## Conclusione
Le ottimizzazioni hanno confermato che il collo di bottiglia principale era nel calcolo interno di `zepyros`. Il riuso di `zernike_obj` ha portato un miglioramento drastico del runtime, mantenendo i risultati numericamente equivalenti agli originals.
