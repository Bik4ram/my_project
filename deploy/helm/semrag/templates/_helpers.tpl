{{/*
Common helpers used across the chart.
*/}}

{{/* Chart name, lowercased. */}}
{{- define "semrag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully-qualified app name. Truncated to 63 chars (the k8s label limit).
*/}}
{{- define "semrag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Chart name + version, used as a label. */}}
{{- define "semrag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Standard recommended labels. */}}
{{- define "semrag.labels" -}}
helm.sh/chart: {{ include "semrag.chart" . }}
{{ include "semrag.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: semrag
{{- with .Values.labels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/* Selector labels — only stable labels that don't change across upgrades. */}}
{{- define "semrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "semrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* ServiceAccount name. */}}
{{- define "semrag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "semrag.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Secret name holding credentials. */}}
{{- define "semrag.secretName" -}}
{{- if .Values.existingSecret -}}
{{- .Values.existingSecret -}}
{{- else -}}
{{- include "semrag.fullname" . -}}
{{- end -}}
{{- end -}}