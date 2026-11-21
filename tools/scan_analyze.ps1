# Analyses an F11 draw-stepper sweep (diagnostics\scan_<stamp>\): mean luminance of the lower part of the scene after
# each step, and the steps with the biggest brightness changes together with the draw that step added.
param(
  [Parameter(Mandatory = $true)][string]$ScanDir,
  [int]$Top = 12,
  [double]$FromY = 0.45,   # fraction of the image height where the measured region starts
  [string]$Csv = ""
)
Add-Type -AssemblyName System.Drawing
$descs = @{}
$txt = Join-Path $ScanDir "scan.txt"
if (Test-Path $txt) {
  foreach ($line in Get-Content $txt) {
    if ($line -match '^limit (\d+): (.*)$') { $descs[[int]$matches[1]] = $matches[2] }
  }
}
$rows = foreach ($f in (Get-ChildItem $ScanDir -Filter "scan_*.bmp" | Sort-Object Name)) {
  $limit = [int]($f.BaseName -replace 'scan_', '')
  $b = [Drawing.Bitmap]::FromFile($f.FullName)
  $sum = 0.0; $n = 0
  for ($y = [int]($b.Height * $FromY); $y -lt $b.Height; $y += 16) {
    for ($x = 0; $x -lt $b.Width; $x += 16) {
      $c = $b.GetPixel($x, $y)
      $sum += 0.2126 * $c.R + 0.7152 * $c.G + 0.0722 * $c.B
      $n++
    }
  }
  $b.Dispose()
  [pscustomobject]@{ Limit = $limit; Luma = [math]::Round($sum / [math]::Max(1, $n), 2) }
}
$prev = $null
$deltas = foreach ($r in $rows) {
  if ($prev) { [pscustomobject]@{ Limit = $r.Limit; From = $prev.Luma; To = $r.Luma; Delta = [math]::Round($r.Luma - $prev.Luma, 2) } }
  $prev = $r
}
if ($Csv) { $rows | Export-Csv -NoTypeInformation $Csv }
"steps: $($rows.Count)   luminance first/last: $($rows[0].Luma) / $($rows[-1].Luma)"
"--- biggest DARKENING steps (draw that was added in the step):"
$deltas | Sort-Object Delta | Select-Object -First $Top | ForEach-Object {
  "limit {0,4}: {1,6} -> {2,6} ({3,6})  {4}" -f $_.Limit, $_.From, $_.To, $_.Delta, $descs[$_.Limit]
}
"--- biggest BRIGHTENING steps:"
$deltas | Sort-Object Delta -Descending | Select-Object -First ([math]::Min(6, $Top)) | ForEach-Object {
  "limit {0,4}: {1,6} -> {2,6} ({3,6})  {4}" -f $_.Limit, $_.From, $_.To, $_.Delta, $descs[$_.Limit]
}
