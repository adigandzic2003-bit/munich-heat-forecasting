from io import BytesIO
import zipfile

import pandas as pd
import pytest

from heat_forecasting.acquisition import archive_catalog, parse_station_zip


def test_catalog_selects_latest_end_date():
    html = ('<a href="tageswerte_KL_03379_19540601_20241231_hist.zip">a</a>'
            '<a href="tageswerte_KL_03379_19540601_20251231_hist.zip">b</a>')
    result = archive_catalog(html, 'https://example.com/climate/')
    assert result['03379'].endswith('20251231_hist.zip')


def test_parse_zip_handles_missing_sentinels():
    content = 'STATIONS_ID;MESS_DATUM;QN_4;TXK;TMK;TNK;eor\n3379;20250101;3;30.2;20;12;eor\n3379;20250102;3;-999;-999;-999;eor\n'
    mem = BytesIO()
    with zipfile.ZipFile(mem, 'w') as z:
        z.writestr('produkt_klima_tag_123.txt', content)
    data = parse_station_zip(mem.getvalue(), '03379')
    assert data.loc[0, 'tmax_c'] == 30.2
    assert pd.isna(data.loc[1, 'tmax_c'])
    with pytest.raises(ValueError, match='station ID'):
        parse_station_zip(mem.getvalue(), '01262')
