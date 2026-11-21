# For an F11 sweep: per step, how many pixels changed versus the previous step (the area that step's draws painted) and
# how bright that new paint is, with the draw that was added. Finds the draws that paint large areas (ground, buildings).
param(
  [Parameter(Mandatory = $true)][string]$ScanDir,
  [int]$Top = 15,
  [int]$Threshold = 2
)
Add-Type -TypeDefinition @"
using System; using System.Drawing; using System.Drawing.Imaging;
public static class ScanDiff {
  public static void Compare(string a, string b, int thr, out double changedFrac, out double meanLuma, out double regionX, out double regionY) {
    using (var A = new Bitmap(a)) using (var B = new Bitmap(b)) {
      var ra = new Rectangle(0,0,A.Width,A.Height);
      var da = A.LockBits(ra, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
      var db = B.LockBits(ra, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
      long n = 0, ch = 0; double sum = 0, sx = 0, sy = 0;
      byte[] pa = new byte[da.Stride * A.Height], pb = new byte[db.Stride * B.Height];
      System.Runtime.InteropServices.Marshal.Copy(da.Scan0, pa, 0, pa.Length);
      System.Runtime.InteropServices.Marshal.Copy(db.Scan0, pb, 0, pb.Length);
      for (int y = 0; y < A.Height; y += 3) for (int x = 0; x < A.Width; x += 3) {
        int i = y * da.Stride + x * 3; n++;
        int d = Math.Abs(pa[i]-pb[i]) + Math.Abs(pa[i+1]-pb[i+1]) + Math.Abs(pa[i+2]-pb[i+2]);
        if (d > thr) { ch++; sum += 0.2126*pb[i+2] + 0.7152*pb[i+1] + 0.0722*pb[i]; sx += x; sy += y; }
      }
      A.UnlockBits(da); B.UnlockBits(db);
      changedFrac = n == 0 ? 0 : (double)ch / n; meanLuma = ch == 0 ? 0 : sum / ch;
      regionX = ch == 0 ? 0 : sx / ch / A.Width; regionY = ch == 0 ? 0 : sy / ch / A.Height;
    }
  }
}
"@ -ReferencedAssemblies System.Drawing 2>$null
$descs = @{}
foreach ($line in Get-Content (Join-Path $ScanDir "scan.txt")) { if ($line -match '^limit (\d+): (.*)$') { $descs[[int]$matches[1]] = $matches[2] } }
$files = Get-ChildItem $ScanDir -Filter "scan_*.bmp" | Sort-Object Name
$rows = @()
for ($i = 1; $i -lt $files.Count; ++$i) {
  [double]$cf = 0; [double]$ml = 0; [double]$rx = 0; [double]$ry = 0
  [ScanDiff]::Compare($files[$i-1].FullName, $files[$i].FullName, $Threshold, [ref]$cf, [ref]$ml, [ref]$rx, [ref]$ry)
  $limit = [int]($files[$i].BaseName -replace 'scan_', '')
  $rows += [pscustomobject]@{ Limit = $limit; Changed = [math]::Round($cf * 100, 2); NewLuma = [math]::Round($ml, 1); CX = [math]::Round($rx, 2); CY = [math]::Round($ry, 2); Desc = $descs[$limit] }
}
"steps: $($rows.Count)"
"--- steps that painted the most screen area (changed %, mean brightness of the new paint, centre x/y as fraction of the screen):"
$rows | Sort-Object Changed -Descending | Select-Object -First $Top | ForEach-Object {
  $d = $_.Desc; if ($d -and $d.Length -gt 300) { $d = $d.Substring(0, 300) + "..." }
  "limit {0,4}: {1,6}% new, luma {2,6}, centre ({3},{4})  {5}" -f $_.Limit, $_.Changed, $_.NewLuma, $_.CX, $_.CY, $d
}
