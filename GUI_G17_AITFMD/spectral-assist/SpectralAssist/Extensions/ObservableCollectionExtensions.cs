using System;
using System.Collections.ObjectModel;
using System.Linq;

namespace SpectralAssist.Extensions;

public static class ObservableCollectionExtensions
{
    extension<T>(ObservableCollection<T> collection)
    {
        public void RemoveAll(Func<T, bool> predicate)
        {
            var itemsToRemove = collection.Where(predicate).ToList();
            foreach (var item in itemsToRemove)
                collection.Remove(item);
        }

        public void Sort(Comparison<T> comparison)
        {
            var sorted = collection.ToList();
            sorted.Sort(comparison);

            collection.Clear();
            foreach (var item in sorted)
                collection.Add(item);
        }
    }
}