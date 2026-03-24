# Doctor Prescription Analyzer using OCR and NLP

## Overview

This project aims to analyze doctor prescriptions by extracting meaningful medical information such as medicine names, dosage, frequency, timing, and duration. It combines Optical Character Recognition (OCR) and Natural Language Processing (NLP) techniques to convert unstructured prescription data into structured, patient-friendly output.

--------------------

## Features
- OCR-based Text Extraction from prescription images
- Text Preprocessing & Cleaning (lowercasing, normalization)
- Regex-based Information Extraction
  - Dosage
  - Frequency
  - Timing
  - Duration
- Medicine Name Detection using dictionary matching
- ML Classification (TF-IDF + Logistic Regression)
  - DO
  - DONT
  - WARNING
- Structured Output Generation

--------------------------

## System Pipeline
Prescription Image
           
OCR (Tesseract)
        
Text Cleaning
        
Regex Extraction
        
Medicine Dictionary Matching
        
TF-IDF Vectorization
        
ML Classification
        
Structured Output
