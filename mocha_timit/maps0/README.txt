MOCHA MultiCHannel Articulatory database: English 


Index
	1. GENERAL DESCRIPTION OF THE MOCHA DATABASE
		1.1 Instrumentation
		1.2 Corpus
		1.3 Subjects
		1.4 Recording Conditions
		1.5 Languages
		1.6 Software
		1.7 Further Information
		1.8 References

	2. STRUCTURE AND DESCRIPTION OF THE DOWNLOAD
		2.1 Language
		2.2 Speakers
		2.3 Download Structure
		2.4 File Naming Convention
		2.5 File Format
		2.6 Compression
		2.7 Information for DOS users
		2.8 Missing and Corrupt Files
		2.9 License agreement.
 
1. GENERAL DESCRIPTION OF THE MOCHA DATABASE
The MOCHA database is compiled as part of the Engineering and Physical
Sciences Research Council grant number:GR/L78680 : "Speech recognition 
using articulatory data."
 
1.1 Instrumentation:
 	The speech material was recorded simultaneously onto three
        computers using serial port communications. The PC-based system 
	was developed as part of a collaborative project between 
	the Department of Speech and Language Sciences at Queen Margaret 
	University College and the Department of Linguistics at the 
	University of Edinburgh and was funded by a Scottish Higher 
	Education Council "Strategic Initiatives Grant".
	The following instrumentation was used:

	Microphone 16kHz sample rate (audio-technica ATM10a)
	Fourcin Laryngograph 16kHz sample rate
	Carstens Articulograph 500Hz sample rate 10 2mm sensors
	Reading Electropalatograph (EPG) 200Hz sample rate

1.2 Corpus:
        A set of 460 short sentences designed to include the main
	connected speech processes in English (eg. assimilations, 
	weak forms ..).
 
1.3 Subjects:
	2 speakers, 1 male and 1 female are currently available but another 38
	are planned to be completed by May 2001. The subjects have a variety
	of accents of English. 
 
1.4 Recording Conditions:
	All recordings made in the same sound damped studio at the Edinburgh 
	Speech Production Facility based in the department of Speech and 
	Language Sciences, Queen Margaret University College, UK. All data 
	were recorded direct to computer and carefully synchronised. 
 
1.5 Languages:
 
        English
 
1.6 Software:
	Edinburgh Speech Tools - File format conversion and Speech
				signal processing functions
       	MATLAB-EMATools  - Graphical Interface for simultaneous
				analysis  of EMA EPG audio and laryngograph data
       	MATLAB-EMATools supplimentary - extra functions to upgrade the
				EMATools suite of macros including routines to 
				read and write the download file formats.
 
1.7 Further information 
Contact:
 
        A. Wrench 
	Dept. Speech and language Sciences, 
	Queen Margaret University College, 
	Clerwood Terrace, 
	Edinburgh. EH12 8TS.
 
	Telephone: +44 131 317 3692
 
	Fax: +44 131 317 3689
 
	Email: a.wrench@sls.qmced.ac.uk
	   or  f.gibbon@sls.qmced.ac.uk
 
	WWW: http://sls.qmced.ac.uk

1.8 References



2. STRUCTURE AND DESCRIPTION OF THIS DOWNLOAD
 Language:
	English

 Speakers:
	msak0	male	Northern English
	fsew0	female	Southern English

2.3 Download Structure:
	Each speaker's files are packed into a single compressed tar
file mocha-timit_<speaker ID> along with a file containing details of
missing or corrupt files and a copy of the license agreement.

e.g.  mocha-timit_msak0.tgz   (gzipped tar file)
	This file may be unpacked using tar -xzf mocha-timit_msak0.tgz
on Unix based platforms. On PC packages Winzip shareware will do the
same job.

2.4 File Naming Convention
			msak0_???.wav audio 				
			msak0_???.lar laryngograph 				
			msak0_???.ema EMA 				
			msak0_???.epg epg (epg3 compatible)

where: ??? is a 3 digit number corresponding to the orthograpic index.

2.5 File Format
	Acoustic Speech Waveform *.wav, 
		16bit 16kHz sample rate binary files with NIST headers. 
	Laryngograph Waveform    *.lar, 
		16bit 16kHz sample rate binary files with NIST headers. 
	Electromagnetic Articulograph	*.ema
		16bit (stored as 4byte float) 500Hz binary files with
		EST headers. Edinburgh Speech Tools Trackfile format 
		consists of a variable length ascii header and a 4 byte 
		float representation per channel. The first channel is a 
		time value in seconds the second value is always 1 (used 
		to indicate if the sample is present or not) subsequent 5 
		values are coil 1-5 x-values followed by coil 1-5 y-values 
		followed by coil 6-10 x-values and finally coils 6-10 y-values. 
	Electropalatograph Frames *.epg, 
		62bit padded to 64bit, Raw binary. 
		Frame rate of 200 frames per second. Each
		frame consists of 12x8bit words. IGNORING THE FIRST 4 WORDS, 
		each bit of each word represents the on/off status of each 
		contact in the palatogram. The first word represents, left 
		to right, the front row of the palatogram (bits 0 and 7 are 
		unused), the last word represents the back row.

2.6 Compression:
	No compression is applied to the files.

2.7 Information for Windows users:
	   Windows NT/95 ports of the Edinburgh Speech Tools now exist 
	using the Cygnus GNU win32 suite, and to a certain extent Visual C++.  
	They seem to work but probably require more work.
	EMAtools works under the Windows version 5.3 or higher of MATLAB

2.8 Missing and corrupt Files
	The accompanying file orth_Eng.txt contains the orthography
	for a single repetition of the data as spoken by all of the subjects.
	The file README.<speaker name> contains details of missing or
	corrupt files.

2.9 Licence Agreement
        A copy of the Licence is contained in the file LICENCE.txt and must be
	kept with the data.
/*************************************************************************/
/*                                                                       */
/*               Department of Speech and Language Sciences              */
/*                 Queen Margaret University College, UK                 */
/*                       Copyright (c) 1999                              */
/*                       All Rights Reserved.                            */
/*                                                                       */
/*  Permission to use, copy, modify, distribute this data and its        */
/*  documentation for research, educational and individual use only, is  */
/*  hereby granted without fee, subject to the following conditions:     */
/*   1. This licence file containing the copyright notice, this list of  */
/*      conditions and the following disclaimer must be retained with    */
/*      the data files.                                                  */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*  This data may not be used for commercial purposes without            */
/*  specific prior written permission from the authors.                  */
/*                                                                       */
/*  QUEEN MARGARET UNIVERSITY COLLEGE AND THE CONTRIBUTORS TO THIS WORK  */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS DATA, INCLUDING          */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL QUEEN MARGARET UNIVERSITY COLLEGE NOR THE CONTRIBUTORS BE      */
/*  LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY     */
/*  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,      */
/*  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS       */
/*  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OF THIS DATA    */
/*                                                                       */
/*************************************************************************/
	
