using System.IO;

namespace SpectralAssist.Utils;

public class PathHelper
{
    public static string ResolveMacResourceFork(string filePath)
    {
        if (!Path.GetFileName(filePath).StartsWith("._")) return filePath;
        var dir = Path.GetDirectoryName(filePath);
        var actualName = Path.GetFileName(filePath)[2..];
        var actualPath = !string.IsNullOrEmpty(dir) ? Path.Combine(dir, actualName) : actualName;
        return File.Exists(actualPath) ? actualPath : filePath;
    }
}