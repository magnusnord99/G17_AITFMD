using System.Collections.Generic;
using System.Linq;
using SpectralAssist.Models;
using SpectralAssist.Services.Library;

namespace SpectralAssist.Extensions;

public static class FolderNodeExtension
{
    extension(FolderNode folder)
    {
        public IEnumerable<ImageNode> AllImages() =>
            LibraryScanner.FlattenImages(new[] { folder });

        public int TotalImageCount() =>
            folder.AllImages().Count();

        public int TotalReportCount() =>
            folder.AllImages().Sum(i => i.Runs.Count);
    }
}