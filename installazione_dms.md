# Guida Definitiva all'installazione di DMS su WSL / Ubuntu

Questa guida permette di compilare e installare `dms` (un software in K&R C del 2002) sui moderni sistemi Linux, aggirando la rigidità dei compilatori attuali (errori di funzioni implicite) e correggendo i difetti del vecchio Makefile originale.

---

## 1. Installazione dei prerequisiti

Aggiorna i pacchetti di sistema e assicurati di avere gli strumenti di compilazione essenziali e l'utility di decompressione.

```bash
sudo apt update
sudo apt install -y unzip build-essential
```

---

## 2. Estrazione del codice sorgente

Posiziona il file `dms.zip` nella cartella di lavoro, per esempio `~/projects/`, estrailo ed entra nella directory generata:

```bash
unzip dms.zip
cd dms_directory_creata
```

Sostituisci `dms_directory_creata` con il nome esatto della cartella.

---

## 3. Compilazione del Client e del Server

I compilatori moderni bloccano il vecchio codice K&R C. Dobbiamo forzare lo standard `gnu89` e passare esplicitamente i percorsi di sistema, altrimenti il client non troverà mai il suo demone.

### Compila il Client (`dms`)

```bash
make clean
make LIBDIR=/usr/local/lib/dms OPT="-O -std=gnu89 -Wno-implicit-function-declaration -Wno-incompatible-pointer-types"
```

### Compila il Server (`dmsd`)

```bash
cd dmsd
make clean
make LIBDIR=/usr/local/lib/dms OPT="-O -std=gnu89 -Wno-implicit-function-declaration -Wno-incompatible-pointer-types"
cd ..
```

---

## 4. Creazione dell'albero delle directory

Il Makefile originale fallisce se le cartelle di destinazione non sono già presenti nel sistema. Creiamole a mano:

```bash
sudo mkdir -p /usr/local/bin
sudo mkdir -p /usr/local/lib/dms/dms
sudo mkdir -p /usr/local/man/man1
```

---

## 5. Copia manuale dei file nel sistema

Questa è la fase cruciale in cui inseriamo il software nel "cuore" di Linux per renderlo accessibile ovunque.

### Copia l'eseguibile principale

```bash
sudo cp dms /usr/local/bin/
```

### Copia il server e rendilo eseguibile

```bash
sudo cp dmsd/dmsd /usr/local/lib/dms/dms/
sudo chmod +x /usr/local/lib/dms/dms/dmsd
```

### Copia il file dei raggi di Van der Waals

Attenzione: il file si chiama `radii.proto` nel sorgente, ma il programma esige che venga rinominato in `radii` senza estensione.

```bash
sudo cp radii.proto /usr/local/lib/dms/dms/radii
```

---

## 6. Pulizia e Test Finale

Una volta installato tutto nei percorsi globali di root, i file sorgente non servono più.

```bash
cd ..
rm -rf dms_directory_creata
rm dms.zip
```

Per confermare il successo dell'operazione, digita il comando da qualsiasi cartella:

```bash
dms
```

Se compare:

```text
usage: dms [options] pdb_file
```

l'installazione è andata a buon fine.

---

## Nota per VS Code

Se apri questo file con VS Code, puoi cliccare sull'icona in alto a destra **Open Preview to the Side** per vedere l'anteprima Markdown formattata.
