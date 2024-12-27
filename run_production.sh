#!/bin/bash
# coding=utf-8
gunicorn --workers=4 --threads=4 -b localhost:56088 app:app