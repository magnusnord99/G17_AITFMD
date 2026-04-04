using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Hsi;

public static class HsiHeaderParser
{
    public static HsiHeader Parse(string hdrPath)
    {
        var resolvedPath = ResolveMacResourceForkPath(hdrPath);
        var fields = ParseFields(resolvedPath);

        var header = new HsiHeader
        {
            DataFilePath = ResolveDataFilePath(resolvedPath),
            Samples = GetInt(fields, "samples"),
            Lines = GetInt(fields, "lines"),
            Bands = GetInt(fields, "bands"),
            HeaderOffset = GetInt(fields, "header offset", 0),
            DataType = GetInt(fields, "data type"),
            ByteOrder = GetInt(fields, "byte order", 0),
            Interleave = GetString(fields, "interleave"),
            Description = GetString(fields, "description", ""),
            WavelengthUnit = GetString(fields, "wavelength units", ""),
        };
        
        if (fields.TryGetValue("default bands", out var db))
            header.DefaultBands = ParseIntList(db);

        if (fields.TryGetValue("wavelength", out var wl))
            header.WavelengthValues = ParseFloatList(wl);
        
        return header;
    }
    
    private static Dictionary<string, string> ParseFields2(string text)
    {
        var fields = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var lines = text.Split(["\r\n", "\n", "\r"], StringSplitOptions.None);

        var i = 1; // skips top "ENVI" line
        while (i < lines.Length)
        {
            var line = lines[i].Trim();
            var equalsIndex = line.IndexOf('=');
            if (equalsIndex < 0) { i++; continue; }
            
            var key = line[..equalsIndex].Trim();
            var value = line[(equalsIndex + 1)..].Trim();
            
            // Handles multi-line blocks
            if (value.Contains('{') && !value.Contains('}'))
            {
                var sb = new StringBuilder(value);
                while (++i < lines.Length)
                {
                    sb.Append(' ').Append(lines[i].Trim());
                    if (lines[i].Contains('}')) break;
                }
                value = sb.ToString();
            }

            // Strip outer braces
            if (value.StartsWith('{') && value.EndsWith('}'))
                value = value[1..^1].Trim();
            
            fields[key] = value;
            i++;
        }

        return fields;
    }
    
    /// <summary>
    /// On macOS, paths may point to ._ resource-fork files. Resolve to the actual ENVI file.
    /// Also handles file:// URLs.
    /// </summary>
    private static string ResolveMacResourceForkPath(string hdrPath)
    {
        var path = hdrPath;
        if (path.StartsWith("file://", StringComparison.OrdinalIgnoreCase) && Uri.TryCreate(path, UriKind.Absolute, out var uri))
            path = uri!.LocalPath;

        var name = Path.GetFileName(path);
        if (string.IsNullOrEmpty(name) || !name.StartsWith("._", StringComparison.Ordinal))
            return path;

        var dir = Path.GetDirectoryName(path);
        var actualName = name.Substring(2);
        var actualPath = !string.IsNullOrEmpty(dir) ? Path.Combine(dir, actualName) : actualName;

        return File.Exists(actualPath) ? Path.GetFullPath(actualPath) : path;
    }

    private static Dictionary<string, string> ParseFields(string hdrPath)
    {
        var fields = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        using var reader = new StreamReader(hdrPath);
        
        while (reader.ReadLine() is { } line)
        {
            line = line.Trim();
            var equalsIndex = line.IndexOf('=');
            if (equalsIndex < 0) continue;

            var key = line[..equalsIndex].Trim();
            var value = line[(equalsIndex + 1)..].Trim();

            // Handles multi-line blocks
            if (value.Contains('{') && !value.Contains('}'))
            {
                var sb = new StringBuilder(value);
                while ((line = reader.ReadLine()) != null)
                {
                    sb.Append(' ').Append(line.Trim());
                    if (line.Contains('}')) break;
                }
                value = sb.ToString();
            }

            // Strip outer braces
            if (value.StartsWith('{') && value.EndsWith('}'))
                value = value[1..^1].Trim();

            fields[key] = value;
        }

        return fields;
    }
    
    
    private static string? ResolveDataFilePath(string hdrPath)
    {
        var dir = Path.GetDirectoryName(hdrPath);
        var baseName = Path.GetFileNameWithoutExtension(hdrPath);

        var noExtension = Path.Combine(dir, baseName);
        if (File.Exists(noExtension)) return noExtension;

        string[] extensions = [".raw", ".dat", ".img", ".bin"];
        return extensions.Select(ext => Path.Combine(dir, baseName + ext)).FirstOrDefault(File.Exists);
    }
    
    private static int GetInt(Dictionary<string, string> fields, string key, int? fallback = null)
    {
        if (fields.TryGetValue(key, out var value))
            return int.Parse(value.Trim());

        return fallback ?? throw new FormatException($"ENVI header missing required field: '{key}'");
    }
    
    private static string GetString(Dictionary<string, string> fields, string key, string? fallback = null)
    {
        if (fields.TryGetValue(key, out var value))
            return value.Trim();

        return fallback ?? throw new FormatException($"ENVI header missing required field: '{key}'");
    }
    
    private static int[] ParseIntList(string value) =>
        value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(int.Parse)
            .ToArray();

    private static float[] ParseFloatList(string value) =>
        value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(s => float.Parse(s, CultureInfo.InvariantCulture))
            .ToArray();
}