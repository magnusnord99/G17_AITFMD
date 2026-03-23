using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public interface IClassifier
{
    Task<ClassificationResult> ClassifyImageAsync(HsiCube cube);
}